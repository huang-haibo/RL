import random

import gym
import numpy as np
import torch
from random import sample
import matplotlib.pyplot as plt


class Conf:
    seed = 5
    # eposides = 1000
    # step_max = 100
    test_times = 10
    state_size = 4
    action_size = 1
    step = 50000
    batch_size = 32
    buffer_size = 64
    discount = 0.9

    greedy_epsilon = 0.15
    greedy_discount = 0.999

    target_replace = 30


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(8, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class A2C(object):
    def __init__(self):
        self.conf = Conf()
        self.critic = Critic()
        self.actor = Actor()

        self.learn_step_counter = 0  # 学习步数计数器
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters())
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)
        self.loss_func_actor = torch.nn.NLLLoss(reduction="none")  # 优化器和损失函数

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((self.conf.buffer_size, self.conf.state_size*2+self.conf.action_size+1+1))

        self.loss_set = []
        self.len_ep = []

    def choose_action(self, x, train=True):

        if train:

            if np.random.uniform() > self.conf.greedy_epsilon:
                a = torch.argmax(self.actor(x)).data.numpy()
            else:
                a = np.random.randint(0, 2)

            self.conf.greedy_epsilon *= self.conf.greedy_discount

        else:
            a = torch.argmax(self.actor(x)).data.numpy()

        return a

    def store_transition(self, s, a, r, s_, d):
        transition = np.hstack((s, [a, r], s_, [d]))
        index = self.memory_counter % self.conf.buffer_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.conf.buffer_size, self.conf.batch_size)
        memory_batch = self.memory[sample_index, :]
        # print(memory_batch)
        s_batch = torch.FloatTensor(memory_batch[:, :self.conf.state_size])
        a_batch = torch.LongTensor(memory_batch[:, self.conf.state_size:self.conf.state_size+self.conf.action_size])\
            .squeeze()
        r_batch = torch.FloatTensor(memory_batch[:, self.conf.state_size+self.conf.action_size])
        s_next_batch = torch.FloatTensor(memory_batch[:, -self.conf.state_size-1:-1])
        d_batch = torch.FloatTensor(memory_batch[:, -1])

        v_value = self.critic(s_batch).squeeze()
        v_value_next = self.critic(s_next_batch).squeeze().detach()
        td_error = r_batch + self.conf.discount * (1-d_batch) * v_value_next-v_value

        a_prob = self.actor(s_batch)
        actor_cross = self.loss_func_actor(torch.log(a_prob), a_batch)
        actor_loss = torch.mean(actor_cross*td_error)
        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_actor.step()

        critic_loss = torch.mean(torch.square(td_error))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()


if __name__ == '__main__':

    conf = Conf()

    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(conf.seed)

    a2c_dis = A2C()

    obs = env.reset(seed=conf.seed)

    steps_test = []

    def return_test():
        steps_list = []
        for i in range(conf.test_times):
            test_obs = env.reset()
            test_done = False
            test_step = 0
            while not test_done:
                test_action = a2c_dis.choose_action(torch.tensor(test_obs), train=False)
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
            steps_list.append(test_step)
        # print(np.mean(steps_list))
        return np.mean(steps_list)

    for step in range(conf.step):

        action = a2c_dis.choose_action(torch.tensor(obs))

        obs_next, reward, done, info = env.step(action)

        if reward < 0:
            reward = 0

        a2c_dis.store_transition(obs, action, reward, obs_next, done)

        if a2c_dis.memory_counter > conf.buffer_size:
            a2c_dis.learn()

        if done:
            obs = env.reset()
        else:
            obs = obs_next

        if step > conf.buffer_size and step % 500 == 1:
            step_mean = return_test()
            print(step_mean)
            steps_test.append(step_mean)

    print(steps_test)
    plt.plot(range(len(steps_test)), steps_test)
    plt.show()


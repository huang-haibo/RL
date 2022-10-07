import gym
import numpy as np
import torch
from random import sample
import matplotlib.pyplot as plt


class Conf:
    # eposides = 1000
    # step_max = 100
    test_times = 10
    state_size = 17
    action_size = 6
    step = 40000
    batch_size = 32
    buffer_size = 64
    discount = 0.8
    epsilon = 0.9
    target_replace = 30

    action_bound = 1
    test_max_step = 200


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = torch.nn.Linear(17, 32)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = torch.nn.Linear(17, 32)
        self.relu1 = torch.nn.ReLU()

        self.linear_mu = torch.nn.Linear(32, 6)
        self.tanh = torch.nn.Tanh()
        self.action_bound = 1

        self.sigma_log_min = -20
        self.sigma_log_max = 2

        self.sigma_log = torch.nn.Parameter(torch.zeros(6))

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        mu = self.linear_mu(x)
        mu = self.tanh(mu)*self.action_bound

        # print(self.sigma_log.shape)
        # print(mu.shape)

        # sigma_log = torch.clip(self.sigma_log, self.sigma_log_min, self.sigma_log_max)
        # sigma = torch.exp(sigma_log).expand_as(mu)

        sigma = torch.exp(self.sigma_log).expand_as(mu)

        norm_dist = torch.distributions.normal.Normal(mu, sigma)

        return norm_dist


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
        self.memory = np.zeros((self.conf.buffer_size, self.conf.state_size*2+self.conf.action_size+1))

        self.loss_set = []
        self.len_ep = []

    def choose_action(self, x):
        # print(x)
        a = self.actor(x).sample().data.numpy()
        a = np.clip(a, -self.conf.action_bound, self.conf.action_bound)

        # if np.random.uniform() < self.conf.epsilon:
        #     a = torch.argmax(self.eval_net(x)).data.numpy()
        # else:
        #     a = np.random.randint(0, 2)
        # print(a)
        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
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
        s_next_batch = torch.FloatTensor(memory_batch[:, -self.conf.state_size:])

        v_value = self.critic(s_batch).squeeze()
        v_value_next = self.critic(s_next_batch).squeeze().detach()

        adv = r_batch + self.conf.discount * v_value_next-v_value

        norm_dist = self.actor(s_batch)
        a_log_prob = torch.sum(norm_dist.log_prob(a_batch), dim=-1)
        a_prob = torch.exp(a_log_prob)

        actor_loss = -torch.mean(a_prob*adv)
        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_actor.step()

        critic_loss = torch.mean(torch.square(adv))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()


if __name__ == '__main__':

    env = gym.make("HalfCheetah-v4")
    # env.action_space.seed(42)

    a2c = A2C()

    conf = Conf()

    # obs = env.reset(seed=42)
    obs = env.reset()

    steps_test = []

    reward_test = []

    def return_test():
        steps_list = []
        reward_list = []
        for i in range(conf.test_times):
            test_obs = env.reset()
            test_done = False
            test_step = 0
            reward_sum = 0
            while not test_done and test_step < conf.test_max_step:
                test_action = a2c.choose_action(torch.FloatTensor(test_obs))
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
                reward_sum += test_reward
            steps_list.append(test_step)
            reward_list.append(reward_sum)
        # print(np.mean(steps_list))
        return np.mean(steps_list), np.mean(reward_list)

    for step in range(conf.step):

        action = a2c.choose_action(torch.FloatTensor(obs))

        obs_next, reward, done, info = env.step(action)
        # if reward < 0:
        #     reward = 0
        a2c.store_transition(obs, action, reward, obs_next)

        if a2c.memory_counter > conf.buffer_size:
            a2c.learn()

        if done:
            obs = env.reset()
        else:
            obs = obs_next

        if step > conf.buffer_size and step % 200 == 0:
            step_mean, reward_mean = return_test()
            print(step_mean, reward_mean)
            steps_test.append(step_mean)
            reward_test.append(reward_mean)

    print(reward_test)
    plt.plot(range(len(reward_test)), reward_test)
    plt.show()


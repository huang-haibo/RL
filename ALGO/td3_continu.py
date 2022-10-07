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
    step = 50000
    batch_size = 32
    buffer_size = 64
    discount = 0.8
    epsilon = 0.9
    target_replace = 50
    actor_update = 3
    test_max_step = 200
    soft_tau = 0.1
    action_bound = 1

    explore_noise = 5

    policy_noise = 0.2
    noise_clip = 0.5


# class Critic(torch.nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.linear1 = torch.nn.Linear(17+6, 32)
#         self.relu1 = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(32, 1)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu1(x)
#         x = self.linear2(x)
#         return x

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = torch.nn.Linear(17, 32)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(6, 32)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x_state = x[:, :17]
        x_action = x[:, 17:]

        x_state = self.relu1(self.linear1(x_state))
        x_action = self.relu2(self.linear2(x_action))

        out = self.linear3(x_state+x_action)

        return out


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = torch.nn.Linear(17, 32)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(32, 6)
        self.tanh = torch.nn.Tanh()
        self.action_bound = 1

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.tanh(x)*self.action_bound
        return x


class TD3(object):
    def __init__(self):
        self.conf = Conf()
        self.critic_eval_1 = Critic()
        self.critic_eval_2 = Critic()
        self.critic_target_1 = Critic()
        self.critic_target_2 = Critic()
        self.actor_eval = Actor()
        self.actor_target = Actor()

        self.critic_target_1.load_state_dict(self.critic_eval_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_eval_2.state_dict())
        self.actor_target.load_state_dict(self.actor_eval.state_dict())

        self.learn_step_counter = 0  # 学习步数计数器
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_eval_1.parameters())
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_eval_2.parameters())
        self.optimizer_actor = torch.optim.Adam(self.actor_eval.parameters())
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)
        # self.loss_func_actor = torch.nn.NLLLoss(reduction="none")  # 优化器和损失函数

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((self.conf.buffer_size, self.conf.state_size*2+self.conf.action_size+1))

        self.loss_set = []
        self.len_ep = []

    def choose_action(self, x):
        # print(x)
        a = self.actor_eval(x).data.numpy()

        # if np.random.uniform() < self.conf.epsilon:
        #     a = torch.argmax(self.eval_net(x)).data.numpy()
        # else:
        #     a = np.random.randint(0, 2)
        # print(a)
        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # print(transition)
        index = self.memory_counter % self.conf.buffer_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, target, source):
        for target_param,  source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1-self.conf.soft_tau)*target_param+self.conf.soft_tau*source_param)

    def learn(self):
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.conf.buffer_size, self.conf.batch_size)
        memory_batch = self.memory[sample_index, :]
        # print(memory_batch)
        s_batch = torch.FloatTensor(memory_batch[:, :self.conf.state_size])
        a_batch = torch.FloatTensor(memory_batch[:, self.conf.state_size:self.conf.state_size+self.conf.action_size])
        r_batch = torch.FloatTensor(memory_batch[:, self.conf.state_size+self.conf.action_size])
        s_next_batch = torch.FloatTensor(memory_batch[:, -self.conf.state_size:])

        a_next_batch = self.actor_target(s_next_batch).detach()
        # a_next_noise_batch = torch.distributions.normal.Normal(torch.zeros_like(a_next_batch),
        #                                                        torch.ones_like(a_next_batch)*conf.policy_noise)
        # a_next_noise_batch = a_next_noise_batch.sample()
        a_next_noise_batch = torch.clip(torch.randn_like(a_next_batch)*self.conf.policy_noise,
                                        -conf.noise_clip, conf.noise_clip)
        a_next_batch = torch.clip(a_next_batch+a_next_noise_batch, -conf.action_bound, conf.action_bound)

        q_value_1 = self.critic_eval_1(torch.cat((s_batch, a_batch), dim=-1)).squeeze()
        q_value_2 = self.critic_eval_2(torch.cat((s_batch, a_batch), dim=-1)).squeeze()

        q_value_next_1 = self.critic_target_1(torch.cat((s_next_batch, a_next_batch), dim=-1)).squeeze().detach()
        q_value_next_2 = self.critic_target_2(torch.cat((s_next_batch, a_next_batch), dim=-1)).squeeze().detach()
        q_value_next_minimum = torch.minimum(q_value_next_1, q_value_next_2)

        td_error_1 = r_batch + self.conf.discount * q_value_next_minimum-q_value_1
        td_error_2 = r_batch + self.conf.discount * q_value_next_minimum-q_value_2

        if self.learn_step_counter % self.conf.actor_update == 0:
            a = self.actor_eval(s_batch)
            q_1 = -self.critic_eval_1(torch.cat((s_batch, a), dim=-1)).squeeze()
            actor_loss = torch.mean(q_1)
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        if self.learn_step_counter % self.conf.target_replace == 0:
            self.soft_update(self.critic_target_1, self.critic_eval_1)
            self.soft_update(self.critic_target_2, self.critic_eval_2)
            self.soft_update(self.actor_target, self.actor_eval)

        critic_loss_1 = torch.mean(torch.square(td_error_1))
        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        critic_loss_2 = torch.mean(torch.square(td_error_2))
        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()


if __name__ == '__main__':

    env = gym.make("HalfCheetah-v4")
    # env.action_space.seed(42)

    td3 = TD3()

    # action_var = 3

    conf = Conf()

    # obs = env.reset(seed=42)
    obs = env.reset()

    steps_test = []

    reward_test = []

    explore_noise = conf.explore_noise

    def return_test():
        steps_list = []
        reward_list = []
        for i in range(conf.test_times):
            test_obs = env.reset()
            test_done = False
            test_step = 0
            reward_sum = 0
            while not test_done and test_step < conf.test_max_step:
                test_action = td3.choose_action(torch.FloatTensor(test_obs))
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
                reward_sum += test_reward
            steps_list.append(test_step)
            reward_list.append(reward_sum)
        # print(np.mean(steps_list))
        return np.mean(steps_list), np.mean(reward_list)

    for step in range(conf.step):

        explore_noise *= 0.999

        action = td3.choose_action(torch.FloatTensor(obs))

        action = np.clip(action+np.random.normal(0, explore_noise, size=conf.action_size),
                         -conf.action_bound, conf.action_bound)

        obs_next, reward, done, info = env.step(action)

        # if reward < 0:
        #     reward = 0

        # print(obs, obs_next, action, reward)
        td3.store_transition(obs, action, reward, obs_next)

        if td3.memory_counter > conf.buffer_size:
            td3.learn()

        if done:
            obs = env.reset()
        else:
            obs = obs_next

        if step > conf.buffer_size and step % 200 == 1:
            step_mean, reward_mean = return_test()
            print(step_mean, reward_mean)
            steps_test.append(step_mean)
            reward_test.append(reward_mean)

    print(reward_test)
    plt.plot(range(len(reward_test)), reward_test)
    plt.show()


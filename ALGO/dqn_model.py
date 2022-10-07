import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import sample


class Conf:
    state_size = 4
    action_size = 1
    step = 50000
    batch_size = 32
    buffer_size = 64
    discount = 0.9
    epsilon = 0.9
    target_replace = 30

    test_times = 10


class Q_func(torch.nn.Module):
    def __init__(self):
        super(Q_func, self).__init__()
        self.linear1 = torch.nn.Linear(4, 16)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class DQN(object):
    def __init__(self):
        self.conf = Conf()
        self.eval_net = Q_func()
        self.target_net = Q_func()

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((self.conf.buffer_size, self.conf.state_size*2+self.conf.action_size+1))  # 初始化记忆库
        # 记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters())
        self.loss_func = torch.nn.MSELoss()  # 优化器和损失函数

        self.loss_set = []
        self.reward_set = []
        self.q_set = []

    def choose_action(self, x):

        if np.random.uniform() < self.conf.epsilon:
            a = torch.argmax(self.eval_net(x)).data.numpy()
        else:
            a = np.random.randint(0, 2)

        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.conf.buffer_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新,每隔TARGET_REPLACE_ITE更新一下
        if self.learn_step_counter % self.conf.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # targetnet是时不时更新一下，evalnet是每一步都更新

        sample_index = np.random.choice(self.conf.buffer_size, self.conf.batch_size)
        memory_batch = self.memory[sample_index, :]
        # print(memory_batch)
        s_batch = torch.FloatTensor(memory_batch[:, :self.conf.state_size])
        a_batch = torch.LongTensor(memory_batch[:, self.conf.state_size:self.conf.state_size+self.conf.action_size])
        r_batch = torch.FloatTensor(memory_batch[:, self.conf.state_size+self.conf.action_size])
        s_next_batch = torch.FloatTensor(memory_batch[:, -self.conf.state_size:])

        q_eval = self.eval_net(s_batch).gather(1, a_batch).squeeze()

        q_value_next = self.target_net(s_next_batch).detach()

        q_target = r_batch + self.conf.discount*q_value_next.max(dim=1)[0]
        # .max返回两个tensor， 最大值和对应index

        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer.step()

        self.reward_set.append(torch.mean(r_batch).data)
        self.q_set.append(torch.mean(q_eval).data)
        self.loss_set.append(loss.data)


if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    env.action_space.seed(42)

    dqn = DQN()

    conf = Conf()

    obs = env.reset(seed=42)

    steps = []

    steps_test = []

    def return_test():
        steps_list = []
        for i in range(conf.test_times):
            test_obs = env.reset()
            test_done = False
            test_step = 0
            while not test_done:
                test_action = dqn.choose_action(torch.tensor(test_obs))
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
            steps_list.append(test_step)
        return np.mean(steps_list)

    for step in range(conf.step):

        action = dqn.choose_action(torch.tensor(obs))

        obs_next, reward, done, info = env.step(action)

        dqn.store_transition(obs, action, reward, obs_next)

        if dqn.memory_counter > conf.buffer_size:
            dqn.learn()

        if done:
            obs = env.reset()
        else:
            obs = obs_next

        if step > conf.buffer_size and step % 500 == 1:
            step_mean = return_test()
            print(step_mean)
            steps_test.append(step_mean)


    #     if step > conf.buffer_size and not (step % 50):
    #         print(dqn.reward_set[-1], "  ", dqn.q_set[-1], "  ", dqn.loss_set[-1])
    #
    # result_np = np.array([dqn.reward_set, dqn.q_set, dqn.loss_set]).transpose()
    #
    # np.savetxt("dqn.txt", result_np)

    print(steps_test)
    plt.plot(range(len(steps_test)), steps_test)
    plt.show()

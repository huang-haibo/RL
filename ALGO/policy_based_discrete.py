import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from random import sample


class Conf:
    eposides = 5000
    step_max = 100
    test_times = 10
    state_size = 4
    action_size = 1
    # step = 5000
    # batch_size = 4
    # buffer_size = 32
    discount = 0.8
    epsilon = 0.9
    target_replace = 30


class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
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


class Policy_Discrete(object):
    def __init__(self):
        self.conf = Conf()
        self.policy_net = PolicyNet()

        self.learn_step_counter = 0  # 学习步数计数器
        self.ep_s = []
        self.ep_a = []
        self.ep_r = []
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)
        self.loss_func = torch.nn.NLLLoss(reduction="none")  # 优化器和损失函数

        self.loss_set = []
        self.len_ep = []

    def choose_action(self, x):
        a = torch.argmax(self.policy_net(x)).data.item()

        # if np.random.uniform() < self.conf.epsilon:
        #     a = torch.argmax(self.eval_net(x)).data.numpy()
        # else:
        #     a = np.random.randint(0, 2)

        return a

    def store_transition(self, s, a, r):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)

    def cal_return(self, r):
        ep_return = []
        dis_sum = 0
        for i in range(len(r)):
            dis_sum += self.conf.discount*r[-i-1]
            ep_return.insert(0, dis_sum)

        return ep_return

    def learn(self):
        ep_return = torch.FloatTensor(self.cal_return(self.ep_r))
        # print(self.ep_s)
        ep_s = torch.FloatTensor(np.array(self.ep_s))
        # print(ep_s)
        # print(self.ep_a)
        ep_a = torch.LongTensor(self.ep_a)

        ep_a_prob = self.policy_net(ep_s)
        # print(ep_a_prob)
        # print(ep_a_prob)
        # print(ep_a)
        # print(ep_return)
        cross_loss = self.loss_func(torch.log(ep_a_prob), ep_a)
        # print(cross_loss)
        loss = -torch.mean(cross_loss*ep_return)
        # print(loss)
        # a= c

        # 计算, 更新net
        self.optimizer.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer.step()

        self.loss_set.append(loss.data)


if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    env.action_space.seed(42)

    policy_dis = Policy_Discrete()

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
                test_action = policy_dis.choose_action(torch.tensor(test_obs))
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
            steps_list.append(test_step)
        # print(np.mean(steps_list))
        return np.mean(steps_list)

    for ep in range(conf.eposides):
        policy_dis.ep_s = []
        policy_dis.ep_a = []
        policy_dis.ep_r = []

        for step in range(conf.step_max):

            action = policy_dis.choose_action(torch.tensor(obs))

            obs_next, reward, done, info = env.step(action)

            policy_dis.store_transition(obs, action, reward)

            if done:
                policy_dis.learn()
                obs = env.reset()
                # if ep % 50 == 1:
                #     steps.append(step)
                # # print(step)
                break
            else:
                obs = obs_next

        if ep % 50 == 1:
            step_mean = return_test()
            print(step_mean)
            steps_test.append(step_mean)

    print(steps_test)
    plt.plot(range(len(steps_test)), steps_test)
    plt.show()
    # print(steps)

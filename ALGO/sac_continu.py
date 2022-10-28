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

    soft_tau = 0.1

    action_bound = 1
    test_max_step = 200


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = torch.nn.Linear(17+6, 32)
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

    def act(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        mu = self.linear_mu(x)
        mu = self.tanh(mu) * self.action_bound

        return mu


class SAC(object):
    def __init__(self):
        self.conf = Conf()
        self.critic_eval_1 = Critic()
        self.critic_eval_2 = Critic()
        self.actor_eval = Actor()

        self.critic_target_1 = Critic()
        self.critic_target_2 = Critic()

        self.critic_target_1.load_state_dict(self.critic_eval_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_eval_2.state_dict())

        self.learn_step_counter = 0  # 学习步数计数器
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_eval_1.parameters())
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_eval_2.parameters())
        self.optimizer_actor = torch.optim.Adam(self.actor_eval.parameters())
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)
        # self.loss_func_actor = torch.nn.NLLLoss(reduction="none")  # 优化器和损失函数

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((self.conf.buffer_size, self.conf.state_size*2+self.conf.action_size+1+1))

        self.loss_set = []
        self.len_ep = []

    def choose_action(self, x):
        # print(x)
        a = self.actor_eval(x).sample().data.numpy()
        a = np.clip(a, -self.conf.action_bound, self.conf.action_bound)

        # if np.random.uniform() < self.conf.epsilon:
        #     a = torch.argmax(self.eval_net(x)).data.numpy()
        # else:
        #     a = np.random.randint(0, 2)
        # print(a)
        return a

    def store_transition(self, s, a, r, s_, d):
        transition = np.hstack((s, a, [r], s_, [d]))
        index = self.memory_counter % self.conf.buffer_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.conf.soft_tau) * target_param + self.conf.soft_tau * source_param)

    def learn(self):
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.conf.buffer_size, self.conf.batch_size)
        memory_batch = self.memory[sample_index, :]
        # print(memory_batch)
        s_batch = torch.FloatTensor(memory_batch[:, :self.conf.state_size])
        a_batch = torch.FloatTensor(memory_batch[:, self.conf.state_size:self.conf.state_size+self.conf.action_size])\
            .squeeze()
        r_batch = torch.FloatTensor(memory_batch[:, self.conf.state_size+self.conf.action_size])
        s_next_batch = torch.FloatTensor(memory_batch[:, -self.conf.state_size-1:-1])
        d_batch = torch.FloatTensor(memory_batch[:, -1])

        q_value_1 = self.critic_eval_1(torch.cat((s_batch, a_batch), dim=-1)).squeeze()
        q_value_2 = self.critic_eval_2(torch.cat((s_batch, a_batch), dim=-1)).squeeze()

        norm_dist_next = self.actor_eval(s_next_batch)
        a_next_batch = norm_dist_next.rsample()
        a_next_log_prob = torch.sum(norm_dist_next.log_prob(a_next_batch), dim=-1)
        #看paper
        a_next_log_prob -= torch.sum(2 * (np.log(2)-a_batch-torch.nn.functional.softplus(-2 * a_batch)), dim=1)

        q_value_next_1 = self.critic_eval_1(torch.cat((s_next_batch, a_next_batch), dim=-1)).squeeze()
        q_value_next_2 = self.critic_eval_2(torch.cat((s_next_batch, a_next_batch), dim=-1)).squeeze()
        q_value_next_minimum = torch.minimum(q_value_next_1, q_value_next_2)
        backup = r_batch+self.conf.discount * (1-d_batch) * (q_value_next_minimum - 0.2*a_next_log_prob).detach()

        critic_loss_1 = torch.mean(torch.square(q_value_1 - backup))
        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        critic_loss_2 = torch.mean(torch.square(q_value_2 - backup))
        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()

        norm_dist = self.actor_eval(s_batch)
        a = norm_dist_next.rsample()
        a_log_prob = torch.sum(norm_dist.log_prob(a), dim=-1)
        #看paper
        a_log_prob -= torch.sum(2 * (np.log(2) - a_batch - torch.nn.functional.softplus(-2 * a_batch)), dim=1)
        q_1 = self.critic_eval_1(torch.cat((s_batch, a), dim=-1)).squeeze()
        q_2 = self.critic_eval_2(torch.cat((s_batch, a), dim=-1)).squeeze()
        q = torch.minimum(q_1, q_2)

        actor_loss = -torch.mean(q-a_log_prob)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.soft_update(self.critic_target_1, self.critic_eval_1)
        self.soft_update(self.critic_target_2, self.critic_eval_2)


if __name__ == '__main__':

    env = gym.make("HalfCheetah-v4")
    # env.action_space.seed(42)

    sac = SAC()

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
                test_action = sac.choose_action(torch.FloatTensor(test_obs))
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
                reward_sum += test_reward
            steps_list.append(test_step)
            reward_list.append(reward_sum)
        # print(np.mean(steps_list))
        return np.mean(steps_list), np.mean(reward_list)

    for step in range(conf.step):

        action = sac.choose_action(torch.FloatTensor(obs))

        obs_next, reward, done, info = env.step(action)
        # if reward < 0:
        #     reward = 0
        sac.store_transition(obs, action, reward, obs_next, done)

        if sac.memory_counter > conf.buffer_size:
            sac.learn()

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


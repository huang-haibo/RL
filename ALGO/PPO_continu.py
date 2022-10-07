import gym
import numpy as np
import torch
import matplotlib.pyplot as plt


class Conf:
    episodes = 500
    step_max = 200

    state_size = 17
    action_size = 6

    batch_size = 32
    # buffer_size = 32

    discount = 0.8
    epsilon = 0.9

    ppo_epsilon = 0.1
    kl_target = 0.01
    beta = 0.5

    actor_update = 20
    critic_update = 20

    test_times = 10
    test_max_step = 200

    soft_tau = 0.1
    action_bound = 1


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


'''
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = torch.nn.Linear(17, 32)
        self.relu1 = torch.nn.ReLU()

        self.linear_mu = torch.nn.Linear(32, 6)
        self.tanh = torch.nn.Tanh()
        self.action_bound = 1

        self.linear_sigma = torch.nn.Linear(32, 6)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        mu = self.linear_mu(x)
        mu = self.tanh(mu)*self.action_bound

        sigma = self.linear_sigma(x)
        sigma = self.softplus(sigma)

        # if mu[0][0] == 'nan':
        #     print(mu)
        #     print(sigma)
        norm_dist = torch.distributions.normal.Normal(mu, sigma)

        return norm_dist
'''


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
        # self.linear_sigma = torch.nn.Linear(32, 6)
        # self.softplus = torch.nn.Softplus()

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


class PPO(object):
    def __init__(self):
        self.conf = Conf()
        self.critic = Critic()
        self.actor = Actor()
        self.actor_old = Actor()
        self.kl_target = self.conf.kl_target
        self.beta = 0.5

        self.states = []
        self.actions = []
        self.rewards = []
        self.state_next = None

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters())
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.001)
        self.loss_func_critic = torch.nn.MSELoss()
        # self.loss_func_actor = torch.nn.NLLLoss(reduction="none")  # 优化器和损失函数

        self.learn_step_counter = 0  # 学习步数计数器

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
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.state_next = s_

    def store_clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_next = None

    def soft_update(self, target, source):
        for target_param,  source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1-self.conf.soft_tau)*target_param+self.conf.soft_tau*source_param)

    def learn(self):

        # self.soft_update(self.actor_old, self.actor)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1

        s_batch = torch.FloatTensor(self.states)
        a_batch = torch.FloatTensor(self.actions)
        r_batch = torch.FloatTensor(self.rewards)
        state_next = torch.FloatTensor(self.state_next)

        s_next_batch = torch.cat((s_batch[1:, :], torch.unsqueeze(state_next, dim=0)), dim=0)

        v_target = r_batch+self.conf.discount*self.critic(s_next_batch).squeeze().detach()

        v = self.critic(s_batch).squeeze().detach()

        adv = v_target-v

        norm_dist_old = self.actor_old(s_batch)

        log_prob_action_old = torch.sum(norm_dist_old.log_prob(a_batch), dim=-1).detach()

        for i in range(conf.actor_update):
            # print(s_batch)
            norm_dist = self.actor(s_batch)
            log_prob_action = torch.sum(norm_dist.log_prob(a_batch), dim=-1)
            # print(log_prob_action)

            ratio = torch.exp(log_prob_action-log_prob_action_old)

            actor_loss = ratio*adv
            # # ppo1
            # kl = torch.distributions.kl_divergence(norm_dist, norm_dist_old)
            # kl = torch.mean(kl, dim=-1)
            # actor_loss = -torch.mean(actor_loss-self.beta*kl)
            #
            # if torch.mean(kl) < self.kl_target/1.5:
            #     self.beta /= 2
            # elif torch.mean(kl) > self.kl_target*1.5:
            #     self.beta *= 2

            # ppo2
            actor_loss = torch.minimum(actor_loss, torch.clip(adv, 1-self.conf.ppo_epsilon, 1+self.conf.epsilon)*ratio)
            actor_loss = -torch.mean(actor_loss)

            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.01)
            self.optimizer_actor.step()

        for i in range(conf.critic_update):
            v = torch.squeeze(self.critic(s_batch))
            critic_loss = self.loss_func_critic(v, v_target)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.01)
            self.optimizer_critic.step()


if __name__ == '__main__':

    env = gym.make("HalfCheetah-v4")
    # env.action_space.seed(42)

    ppo = PPO()

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
                test_action = ppo.choose_action(torch.FloatTensor(test_obs))
                test_obs_next, test_reward, test_done, test_info = env.step(test_action)
                test_obs = test_obs_next
                test_step += 1
                reward_sum += test_reward
            steps_list.append(test_step)
            reward_list.append(reward_sum)
        # print(np.mean(steps_list))
        return np.mean(steps_list), np.mean(reward_list)

    for episode in range(conf.episodes):

        step = 0

        done = False

        ppo.store_clear()

        while not done and step < conf.step_max:

            step += 1

            action = ppo.choose_action(torch.FloatTensor(obs))

            obs_next, reward, done, info = env.step(action)

            if reward < 0:
                reward = 0

            # print(obs, obs_next, action, reward)
            ppo.store_transition(obs, action, reward, obs_next)

            if step % conf.batch_size == 0:
                ppo.learn()
                ppo.store_clear()

            if done:
                obs = env.reset()
            else:
                obs = obs_next

        if episode % 10 == 0:
            step_mean, reward_mean = return_test()
            print(step_mean, reward_mean)
            steps_test.append(step_mean)
            reward_test.append(reward_mean)

    print(reward_test)
    plt.plot(range(len(reward_test)), reward_test)
    plt.show()


import gym
from gym import spaces
import numpy as np
import torch
from query.simulate import *
from utils.helpers import reparameterize

class QueryWorld(gym.Env):
    def __init__(self, args, dataset, encoder):
        self.args = args
        self.dataset = dataset
        self.encoder = encoder
        self.horizon = args.sequence_length
        assert self.horizon > 0

    def reset(self):
        self.step_count = 0
        self.state = np.zeros(self.observation_space.shape)
        self.done = False
        self.true_human = self.dataset.get_random_true_rewards(batchsize=1)
        self.hidden = self.encoder.init_hidden(batchsize=1)

    def step(self, action):
        self.step_count += 1
        query = self.dataset.queries[action].unsqueeze(0).clone()
        answer = sample_dist(self.args, response_dist(self.args, query, self.true_human)).squeeze(0)
        mu, logvar, self.hidden = self.encoder(query.unsqueeze(0), self.hidden)
        reward = self.reward_function(query, answer, mu, logvar)
        self.state[:self.args.num_features] = mu.detach().numpy()
        self.state[self.args.num_features:2*self.args.num_features] = logvar.detach().numpy()
        return reward

    def reward_function(self, query, answer, mus, logvars):
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rew = torch.exp(torch.bmm(query, samples.mT))
        denom = torch.sum(rew, dim=-2).unsqueeze(-2)
        posterior = rew/denom
        reward = torch.sum(torch.log2(posterior[:, answer]), dim=-1) - torch.log2(torch.sum(posterior[:, answer], dim=-1))
        return reward.item()

class QueryActionWorld(QueryWorld):

    def __init__(self, *args):
        super().__init__(*args)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2 * self.args.num_features, ))
        self.action_space = spaces.Discrete(self.dataset.buffer_len)
        # set initial vars
        self.reset()

    def reset(self):
        super().reset()
        return self.state

    def step(self, action):
        if self.step_count < self.horizon:
            reward = super().step(action)
        else:
            reward = 0

        done = self.step_count >= self.horizon
        info = {"step_count": self.step_count}

        return self.state, reward, done, info

class QueryStateWorld(QueryWorld):

    def __init__(self, *args):
        super().__init__(*args)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((2+self.args.query_size) * self.args.num_features,))
        self.action_space = spaces.Discrete(2)
        # set initial vars
        self.reset()
    
    def reset(self):
        super().reset()
        query = self.dataset.get_random_queries(batchsize=1).squeeze(0)
        self.state[2*self.args.num_features:] = query.flatten()
        return self.state

    def step(self, action):
        if self.step_count < self.horizon:
            if action == 1:
                reward = super().step(action)
            else:
                reward = 0

            # replace old query in state
            query = self.dataset.get_random_queries(batchsize=1).squeeze(0)
            self.state[2*self.args.num_features:] = query.flatten()
        else:
            reward = 0

        done = self.step_count >= self.horizon
        info = {"step_count": self.step_count}

        return self.state, reward, done, info
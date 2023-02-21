import gym
from gym import spaces
import numpy as np
import torch
import math
from query.simulate import *
from storage.vae_storage import order_queries
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

        self.mu = torch.zeros((1, 1, self.args.num_features))
        self.logvar = torch.zeros((1, 1, self.args.num_features))
        self.state[:self.args.num_features] = self.mu.detach().numpy()
        self.state[self.args.num_features:2*self.args.num_features] = self.logvar.detach().numpy()

    def reward_function(self, query, answer, mus, logvars):
        # TODO change reward formulation
        # # old reward formulation
        # samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        # rew = torch.exp(torch.bmm(query, samples.mT))
        # denom = torch.sum(rew, dim=-2).unsqueeze(-2)
        # posterior = rew/denom
        # reward = torch.mean(torch.log2(posterior[:, answer]), dim=-1) - torch.log2(torch.mean(posterior[:, answer], dim=-1))
        # return reward.item()

        # new reward formulation
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rew = torch.exp(torch.bmm(query, samples.mT))
        denom = torch.sum(rew, dim=-2).unsqueeze(-2)
        posterior = (rew/denom)[:, answer]
        posterior_sum = torch.sum(posterior, dim=-1)
        
        dist = torch.distributions.MultivariateNormal(mus.squeeze(), torch.diag(torch.exp(logvars).squeeze()))
        logprobs = dist.log_prob(samples.squeeze())
        #print(torch.exp(logprobs))
        lognum = (logprobs + torch.log2(posterior.squeeze(1)) - torch.log2(posterior_sum/self.args.m))
        reward = torch.sum(posterior.squeeze(1) * lognum, dim=-1)/posterior_sum - torch.sum(logprobs, dim=-1)/self.args.m
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
            self.step_count += 1
            query = self.dataset.queries[action].unsqueeze(0).clone()
            answer = sample_dist(self.args, response_dist(self.args, query, self.true_human)).squeeze(0)
            reward = self.reward_function(query, answer, self.mu, self.logvar)
            query = order_queries(query, answer)
            self.mu, self.logvar, self.hidden = self.encoder(query.unsqueeze(0), self.hidden)
            self.state[:self.args.num_features] = self.mu.detach().numpy()
            self.state[self.args.num_features:2*self.args.num_features] = self.logvar.detach().numpy()
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
        self.query = self.dataset.get_random_queries(batchsize=1)
        self.state[2*self.args.num_features:] = self.query.squeeze(0).flatten()
        return self.state

    def step(self, action):
        if self.step_count < self.horizon:
            if action == 1:
                self.step_count += 1
                answer = sample_dist(self.args, response_dist(self.args, self.query, self.true_human)).squeeze(0)
                reward = self.reward_function(query, answer, self.mu, self.logvar)
                query = order_queries(self.query, answer)
                self.mu, self.logvar, self.hidden = self.encoder(query.unsqueeze(0), self.hidden)
                self.state[:self.args.num_features] = self.mu.detach().numpy()
                self.state[self.args.num_features:2*self.args.num_features] = self.logvar.detach().numpy()
            else:
                reward = 0

            # replace old query in state
            self.query = self.dataset.get_random_queries(batchsize=1)
            self.state[2*self.args.num_features:] = self.query.squeeze(0).flatten()
        else:
            reward = 0

        done = self.step_count >= self.horizon
        info = {"step_count": self.step_count}

        return self.state, reward, done, info
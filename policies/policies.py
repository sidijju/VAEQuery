import torch
from utils.helpers import reparameterize
from environments import metaenvs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.helpers import makedir
import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from gym import spaces

class Policy:
    def __init__(self, args, encoder, train_dataset):
        self.args = args
        self.encoder = encoder
        self.train_dataset = train_dataset

    def pretrain_policy(self, n):
        pass

    def train_policy(self, i, n=1000):
        pass

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        pass

class RandomPolicy(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis_directory = "random/"

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        return dataset.get_random_queries(self.args.batchsize)

class GreedyPolicy(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis_directory = "greedy/"

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        queries = dataset.queries[:dataset.buffer_len]
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rews = torch.einsum('qij, bmj -> bqim', queries, samples).to(self.args.device)
        rews = torch.exp(rews).to(self.args.device)
        denoms = torch.sum(rews, dim=-2).unsqueeze(-2).to(self.args.device)
        posteriors = rews/denoms
        mutual_answers = []
        for answer in range(self.args.query_size):
            sample_total = torch.sum(posteriors[:, :, answer, :], dim=-1).unsqueeze(-1)
            logmean = torch.log2(self.args.m * posteriors[:, :, answer, :] / sample_total)
            mutual = torch.sum(posteriors[:, :, answer, :] * logmean, dim=-1)
            mutual_answers.append(mutual)
        query_vals = -1.0/self.args.m * torch.sum(torch.stack(mutual_answers).to(self.args.device), dim=0)
        queries_chosen = queries[torch.argmin(query_vals, dim=-1)]
        return queries_chosen

class RLPolicy(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.log_dir = "logs/" + self.args.exp_name + "/" + self.vis_directory
        makedir(self.log_dir)
        self.model, self.env = None, None

    def pretrain_policy(self, n):
        if not self.model:
            self.model = A2C("MlpPolicy", self.env, verbose=1, tensorboard_log=self.log_dir, learning_rate=self.args.lr)
        else:
            self.model.set_env(self.env)
        self.model.learn(total_timesteps=n, log_interval=10, tb_log_name="pre", reset_num_timesteps=True)
        self.model.save(self.log_dir + "/start_model")

    def train_policy(self, i, n):
        if not self.model:
            self.model = A2C("MlpPolicy", self.env, verbose=1, tensorboard_log=self.log_dir, learning_rate=self.args.lr)
        else:
            self.model.set_env(self.env)
        self.model.learn(total_timesteps=n, log_interval=10, tb_log_name="def"+str(i), reset_num_timesteps=False)

class RLStatePolicy(RLPolicy):
    def __init__(self, *args):
        self.vis_directory = "rl/"
        super().__init__(*args)

    # TODO write batched version of this method
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        obs = torch.cat((mus, logvars), dim=-1).cpu().detach().numpy()
        queries = []
        for b in range(mus.shape[0]):
            action, _ = self.model.predict(obs[b])
            queries.append(dataset.queries[action].squeeze(0))
        return torch.stack(queries).squeeze(0).to(self.args.device)
    
    def pretrain_policy(self, n):
        self.env = make_vec_env(lambda: metaenvs.QueryActionWorld(self.args, self.train_dataset, self.encoder, hot_start=True), n_envs=8)
        super().pretrain_policy(n)
    
    def train_policy(self, i, n): 
        self.env = make_vec_env(lambda: metaenvs.QueryActionWorld(self.args, self.train_dataset, self.encoder), n_envs=8)
        super().train_policy(i, n)

class RLFeedPolicy(RLPolicy):
    def __init__(self, *args):
        self.vis_directory = "rl-feed/"
        super().__init__(*args)
        
    # TODO write batched version of this method
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        queries = []
        mus = mus.squeeze(0)
        logvars = logvars.squeeze(0)
        for b in range(mus.shape[0]):
            query = dataset.get_random_queries(batchsize=1).squeeze(0).flatten()
            obs = torch.cat((mus[b], logvars[b], query), dim=-1).cpu().detach().numpy()
            action, _ = self.model.predict(obs)
            while action != 1:
                query = dataset.get_random_queries(batchsize=1).squeeze(0).flatten()
                obs[2*self.args.num_features:] = query.clone().cpu()
                action, _ = self.model.predict(obs)
            queries.append(query.reshape(self.args.query_size,-1))
        return torch.stack(queries).squeeze(0).to(self.args.device)
    
    def pretrain_policy(self, n):
        self.env = make_vec_env(lambda: metaenvs.QueryStateWorld(self.args, self.train_dataset, self.encoder, hot_start=True), n_envs=8)
        super().pretrain_policy(n)

    def train_policy(self, i, n): 
        self.env = make_vec_env(lambda: metaenvs.QueryStateWorld(self.args, self.train_dataset, self.encoder), n_envs=8)
        super().train_policy(i, n)

class GreedyApproximationPolicy(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis_directory = "greedy-approx/"
        self.log_dir = "logs/" + self.args.exp_name + "/" + self.vis_directory
        self.rng = np.random.default_rng(self.args.seed)
        env = metaenvs.QueryActionWorld(self.args, self.train_dataset, self.encoder)
        rollouts = rollout.rollout(
            self.run_greedy_policy,
            DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=1000),
            rng=self.rng,
        )
        transitions = rollout.flatten_trajectories(rollouts)
        self.bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=self.rng,
            policy=MlpPolicy(observation_space=env.observation_space, 
                             action_space=env.action_space, 
                             lr_schedule=lambda _: torch.finfo(torch.float32).max)
        )

    def run_greedy_policy(self, obs) -> torch.Tensor:
        queries = self.train_dataset.queries[:self.train_dataset.buffer_len]
        obs = torch.from_numpy(obs).to(self.args.device)
        mus = obs[:,:self.args.num_features]
        logvars = obs[:,self.args.num_features:]
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rews = torch.einsum('qij, bmj -> bqim', queries, samples).to(self.args.device)
        rews = torch.exp(rews).to(self.args.device)
        denoms = torch.sum(rews, dim=-2).unsqueeze(-2).to(self.args.device)
        posteriors = rews/denoms
        mutual_answers = []
        for answer in range(self.args.query_size):
            sample_total = torch.sum(posteriors[:, :, answer, :], dim=-1).unsqueeze(-1)
            logmean = torch.log2(self.args.m * posteriors[:, :, answer, :] / sample_total)
            mutual = torch.sum(posteriors[:, :, answer, :] * logmean, dim=-1)
            mutual_answers.append(mutual)
        query_vals = -1.0/self.args.m * torch.sum(torch.stack(mutual_answers).to(self.args.device), dim=0)
        return torch.argmin(query_vals, dim=-1).cpu().numpy()
        
    # TODO write batched version of this method
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        obs = torch.cat((mus, logvars), dim=-1).cpu().detach().numpy()
        queries = []
        for b in range(mus.shape[0]):
            action, _ = self.policy.predict(obs[b])
            queries.append(dataset.queries[action].squeeze(0))
        return torch.stack(queries).squeeze(0).to(self.args.device)

    def train_policy(self, i, n):
        self.bc_trainer.train(n_epochs=300)
        self.policy = self.bc_trainer.policy
        self.policy.save(self.log_dir + "model")

class ProbPolicy(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis_directory = "prob/"
        self.log_dir = "logs/" + self.args.exp_name + "/" + self.vis_directory

        # get info gain values for the heuristic queries
        self.heuristic_queries = self.train_dataset.get_random_queries(batchsize=10)
        
    def greedy_heuristic(self, obs):
        mus = obs[:self.args.num_features]
        logvars = obs[self.args.num_features:self.args.num_features*2]
        query = obs[2*self.args.num_features:]
        count = 0
        for heuristic_query in self.heuristic_queries:
            if self.greedy_reward_function(query, mus, logvars) > self.greedy_reward_function(heuristic_query, mus, logvars):
                count += 1
        return count/10.0

    def greedy_reward_function(self, query, mus, logvars):
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rew = torch.exp(torch.bmm(query, samples.mT)).to(self.args.device)
        denom = torch.sum(rew, dim=-2).unsqueeze(-2).to(self.args.device)
        posteriors = rew/denom
        mutual_answers = []
        for answer in range(self.args.query_size):
            sample_total = torch.sum(posteriors[:, answer], dim=-1).unsqueeze(-1)
            logmean = torch.log2(self.args.m * posteriors[:, answer] / sample_total)
            mutual = torch.sum(posteriors[:, answer] * logmean, dim=-1)
            mutual_answers.append(mutual)
        query_val = -1.0/self.args.m * torch.sum(torch.stack(mutual_answers).to(self.args.device), dim=0)
        return query_val.item()

    def run_greedy_policy(self, obs) -> torch.Tensor:
        queries = self.dataset.queries[:self.dataset.buffer_len]
        obs = torch.from_numpy(obs).to(self.args.device)
        mus = obs[:,:self.args.num_features]
        logvars = obs[:,self.args.num_features:]
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rews = torch.einsum('qij, bmj -> bqim', queries, samples).to(self.args.device)
        rews = torch.exp(rews).to(self.args.device)
        denoms = torch.sum(rews, dim=-2).unsqueeze(-2).to(self.args.device)
        posteriors = rews/denoms
        mutual_answers = []
        for answer in range(self.args.query_size):
            sample_total = torch.sum(posteriors[:, :, answer, :], dim=-1).unsqueeze(-1)
            logmean = torch.log2(self.args.m * posteriors[:, :, answer, :] / sample_total)
            mutual = torch.sum(posteriors[:, :, answer, :] * logmean, dim=-1)
            mutual_answers.append(mutual)
        query_vals = -1.0/self.args.m * torch.sum(torch.stack(mutual_answers).to(self.args.device), dim=0)
        return torch.argmin(query_vals, dim=-1).cpu().numpy()
        
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        pass

    def train_policy(self, i, n):
        pass
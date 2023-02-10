import torch
from utils.helpers import reparameterize
from environments import metaenvs
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from utils.helpers import makedir

class Policy:
    def __init__(self, args, encoder):
        self.vis_directory = ""
        self.args = args
        self.encoder = encoder

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        pass

    def train_policy(self, dataset, n=1000):
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

    # TODO write batched version of this method
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        queries = dataset.queries[:dataset.buffer_len]
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        queries_chosen = []
        for i in range(self.args.batchsize):
            rews = torch.exp(torch.matmul(queries, samples[i].T))
            denoms = torch.sum(rews, dim=-2).unsqueeze(-2)
            posteriors = rews/denoms
            # posteriors is P(q | Q, w) for all q, Q, and w
            mutual_answers = []
            for answer in range(self.args.query_size):
                # sum for this answer over all w
                sample_total = torch.sum(posteriors[:, answer, :], dim=-1).unsqueeze(-1)
                assert sample_total.shape == (dataset.buffer_len, 1)
                inter = self.args.m * posteriors[:, answer, :] / sample_total
                log2 = torch.log2(inter)
                mutual = torch.sum(posteriors[:, answer, :] * log2, dim=-1)
                assert mutual.shape == (dataset.buffer_len,)
                mutual_answers.append(mutual)
            query_vals = -1.0/self.args.m * torch.sum(torch.stack(mutual_answers), dim=0)
            queries_chosen.append(queries[torch.argmin(query_vals, dim=-1)])
        return torch.stack(queries_chosen)

class RLPolicy(Policy):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis_directory = "rl/"
        self.log_dir = "logs/" + self.args.exp_name + "/" + self.vis_directory
        makedir(self.log_dir)
        self.model, self.env = None, None

    def train_policy(self, dataset, n):
        if not self.model:
            self.model = A2C("MlpPolicy", self.env, verbose=1, tensorboard_log=self.log_dir, learning_rate=self.args.lr)
        else:
            self.model.set_env(self.env)
        self.model.learn(total_timesteps=n, log_interval=10, progress_bar=True, tb_log_name="def", reset_num_timesteps=False)
        self.model.save(self.log_dir + "/model")

class RLStatePolicy(RLPolicy):
    def __init__(self, *args):
        super().__init__(*args)

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        obs = torch.cat((mus, logvars), dim=-1).detach().numpy()
        queries = []
        for b in range(mus.shape[0]):
            action, _ = self.model.predict(obs[b])
            queries.append(dataset.queries[action].squeeze(0))
        return torch.stack(queries).squeeze(0)

    def train_policy(self, dataset, n): 
        self.env = make_vec_env(lambda: metaenvs.QueryActionWorld(self.args, dataset, self.encoder), n_envs=4)
        super().train_policy(dataset, n)

class RLFeedPolicy(RLPolicy):
    def __init__(self, *args):
        super().__init__(*args)
        self.vis_directory = "rl-feed/"
        self.log_dir = "logs/" + self.args.exp_name + self.vis_directory
        self.model = None
    
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        queries = []
        for b in range(mus.shape[0]):
            query = dataset.get_random_queries(batchsize=1).squeeze(0).flatten()
            obs = torch.cat((mus[b], logvars[b], query), dim=-1).detach().numpy()
            action, _ = self.model.predict(obs)
            while action != 1:
                query = dataset.get_random_queries(batchsize=1).squeeze(0).flatten()
                obs[2*self.args.num_features:] = query
                action, _ = self.model.predict(obs)
            queries.append(query.reshape(self.args.query_size,-1))
        return torch.stack(queries).squeeze(0)

    def train_policy(self, dataset, n): 
        self.env = make_vec_env(lambda: metaenvs.QueryStateWorld(self.args, dataset, self.encoder), n_envs=4)
        super().train_policy(dataset, n)
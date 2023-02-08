import numpy as np
import torch
from utils.helpers import reparameterize

class Policy:
    def __init__(self, args):
        self.vis_directory = ""
        self.args = args

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        pass

    def train_policy(self, dataset, n=1000):
        pass

class RandomPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)
        self.vis_directory = "random/"

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        return dataset.get_random_queries(self.args.batchsize)

class GreedyPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)
        self.vis_directory = "greedy/"

    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        queries = dataset.queries[:dataset.buffer_len].unsqueeze(1)
        # generate samples of w
        samples = reparameterize(self.args, mus, logvars, samples=self.args.m)
        rews = torch.exp(torch.matmul(queries, samples.mT))
        rews = rews.transpose(0, 1)
        denoms = torch.sum(rews, dim=-2).unsqueeze(-2)
        posteriors = rews/denoms
        # posteriors is P(q | Q, w) for all q, Q, and w
        mutual_answers = []
        for answer in range(self.args.query_size):
            # sum for this answer over all w
            sample_total = torch.sum(posteriors[:, :, answer], dim=-1).unsqueeze(-1)
            assert sample_total.shape == (self.args.batchsize, dataset.buffer_len, 1)
            inter = self.args.batchsize * posteriors[:, :, answer] / sample_total
            log2 = torch.log2(inter)
            mutual = torch.sum(posteriors[:, :, answer] * log2, dim=-1)
            assert mutual.shape == (self.args.batchsize, dataset.buffer_len,)
            mutual_answers.append(mutual)
        query_vals = -1.0/self.args.batchsize * torch.sum(torch.stack(mutual_answers), dim=0)
        return queries[torch.argmin(query_vals, dim=-1)].squeeze(1)

class RLPolicy(Policy):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.vis_directory = "rl/"
    
    def run_policy(self, mus, logvars, dataset) -> torch.Tensor:
        raise NotImplementedError

    def train_policy(self, dataset, n=1000):
        raise NotImplementedError
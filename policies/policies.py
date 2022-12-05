import numpy as np
import torch

class Policy:
    def __init__(self, args, dataset):
        self.vis_directory = ""
        self.dataset = dataset
        self.args = args

    def run_policy(self, queries, beliefs) -> torch.Tensor:
        pass

    def train_policy(self, n=1000):
        pass

class RandomPolicy(Policy):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.vis_directory = "random/"

    def run_policy(self, queries, beliefs) -> torch.Tensor:
        return self.dataset.get_random_queries(self.args.batchsize)

class GreedyPolicy(Policy):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.vis_directory = "greedy/"

    def run_policy(self, queries, beliefs) -> torch.Tensor:
        raise NotImplementedError

class RLPolicy(Policy):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.vis_directory = "rl/"
    
    def run_policy(self, queries, beliefs) -> torch.Tensor:
        raise NotImplementedError

    def train_policy(self, n=1000):
        raise NotImplementedError
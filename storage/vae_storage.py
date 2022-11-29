import numpy as np
import torch
from utils.helpers import FeatureExtractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAEStorage:

    # Storage of trajectories for VAE training
    
    def __init__(self, args):

        self.args = args
    
        assert args.num_features > 0

        # max number of trajectories
        self.buffer_size = self.args.buffer_size
        assert self.buffer_size > 0

        self.buffer_len = 0
        self.idx = 0

        self.trajectories = torch.zeros((self.buffer_size, args.num_features))

        self.feature_extractor = FeatureExtractor()
        
    def get_batch(self, batchsize=5):

        size = min(self.buffer_len, batchsize)

        # select random indices for trajectories
        idx = np.random.choice(range(self.buffer_len), size, replace=False)

        # select the rollouts we want
        return self.trajectories[idx, :, :]

    def insert(self, full_trajectory):

        # featurize full trajectory
        featurized = self.feature_extractor(full_trajectory)

        # ring buffer, replace at beginning
        if self.idx >= self.buffer_size:
            self.buffer_len = self.buffer_size
            self.idx = 0
        else:
            self.buffer_len += 1

        # add to larger buffer
        self.trajectories[self.idx] = featurized
        self.idx += 1
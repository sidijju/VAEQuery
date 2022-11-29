import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from visualize import visualize_behavior

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makedir(dirname = "visualizations"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def sample_gaussian(mu, logvar):
    stddev = torch.exp(0.5 * logvar)
    norm_sample = torch.randn_like(stddev)
    return norm_sample.mul(stddev).add(mu)

class FeatureExtractor:

    #return features from a trajectory

    def __init__(self):
        pass

def collect_trajectory():
    pass

def collect_trajectories(n=100):
    pass

def visualize_trajectory():
    pass
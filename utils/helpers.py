import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makedir(dirname = "visualizations"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def sample_gaussian(mu, logvar):
    stddev = torch.exp(0.5 * logvar)
    norm_sample = torch.randn_like(stddev)
    return norm_sample.mul(stddev).add(mu)

class FeatureExtractor(nn.Module):

    def __init__(self, input_size, output_size, activation_function=F.relu):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        return self.activation_function(self.fc(inputs))
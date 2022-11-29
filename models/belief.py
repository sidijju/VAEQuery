import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import FeatureExtractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Belief(nn.Module):

    def __init__(self, args):
        super(Belief, self).__init__()

        layers = []
        curr = args.latent_dim
        for l in args.fc_layers:
            layers.append(nn.Linear(curr, l))
        self.layers = layers
        # final layers to output normal distribution
        self.fc_mu = nn.Linear(curr, args.num_features)
        self.fc_logvar = nn.Linear(curr, args.num_features)

    def forward(self, query):
        
        # run through layers
        output = query
        for l in self.layers:
            output = l(output)

        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)

        return mu, logvar
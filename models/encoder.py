import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import FeatureExtractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, args, input_dim = 2):
        super(Encoder, self).__init__()
        # input_dim  --> size of input representation for each trajectory (number of features)
        # fc_dim --> size of GRU hidden dimension

        # fc layer before passing into GRU units
        # TODO: add option to make multiple layers
        input_dim = args.query_size * input_dim
        self.fc_before = nn.Linear(input_dim, args.vae_fc_dim)

        # RNN functionality
        self.gru = nn.GRU(input_size=args.vae_fc_dim,
                          hidden_size=args.vae_gru_hidden_size,
                          num_layers=args.vae_gru_hidden_layers)

        # fc layers after the RNN
        # TODO: add option to make multiple layers
        self.output = nn.Linear(args.vae_gru_hidden_size, args.vae_latent_dim)

    def forward(self, query):
        
        # run through layers
        output = self.fc_before(input)
        output, _ = self.gru(output, query)
        output = self.output(output)

        return output
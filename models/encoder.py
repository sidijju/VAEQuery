import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        # fc layer before passing into GRU units
        # TODO: add option to make multiple layers
        input_dim = args.query_size * args.num_features
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
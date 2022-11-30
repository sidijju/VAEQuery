import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args

        # fc layer before passing into GRU units
        # TODO: add option to make multiple layers
        self.input_dim = args.query_size * args.num_features
        self.hidden_dim = args.gru_hidden_size
        self.fc_before = nn.Linear(self.input_dim, args.fc_dim)

        # RNN functionality
        self.gru = nn.GRU(input_size=args.fc_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=args.gru_hidden_layers)

        # fc layers after the RNN
        # TODO: add option to make multiple layers
        self.output = nn.Linear(self.hidden_dim, args.latent_dim)

        self.hidden = None

    def forward(self, query):
        # run through layers
        output = self.fc_before(query)
        output, self.hidden = self.gru(output, self.hidden)
        output = self.output(output)
        output = F.relu(output)
        return output

    def init_hidden(self):
        self.hidden = torch.zeros((1, self.args.batchsize, self.hidden_dim))
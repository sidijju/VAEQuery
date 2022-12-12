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
        self.hidden_dim = args.gru_hidden_size

        self.fc_input_query = nn.Linear(args.num_features, args.fc_dim)
        self.fc_input = nn.Linear(args.query_size * args.fc_dim, 32)

        # RNN functionality
        self.gru = nn.GRU(input_size=32,
                          hidden_size=self.hidden_dim,
                          num_layers=args.gru_hidden_layers)

        # fc layers after the RNN
        fc_layers = []
        curr = self.hidden_dim
        for l in args.fc_layers:
            fc_layers.append(nn.Linear(curr, l))
            curr = l
        self.fc_layers = fc_layers

        self.fc_mu = nn.Linear(curr, args.num_features)
        self.fc_logvar = nn.Linear(curr, args.num_features)

    def reparameterize(self, mu, logvar):
        if self.args.sample_belief:
            # put batch dimension first
            mu = mu.transpose(0, 1)
            logvar = logvar.transpose(0, 1)
            # shape is batch, 1, num_features
            std = torch.exp(0.5*logvar).repeat(1, self.args.m, 1)
            eps = torch.randn_like(std)
            belief = eps.mul(std).add_(mu.repeat(1, self.args.m, 1))
            belief = belief / torch.linalg.norm(belief, dim=-1).unsqueeze(-1)
            belief = torch.mean(belief, dim=-2)
        else:
            belief = mu / torch.linalg.norm(mu, dim=-1).unsqueeze(-1)
        return belief

    def forward(self, query, hidden):
        # embed query inputs
        embeddings = self.fc_input_query(query)
        embeddings = torch.flatten(embeddings, start_dim=-2)

        # fc layer
        output = self.fc_input(embeddings)
        output = F.leaky_relu(output)

        # run through gru
        output, hidden = self.gru(output, hidden)

        # run through fc layers
        for l in self.fc_layers:
            output = l(output)
            output = F.leaky_relu(output)

        # output belief distribution and belief sample
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        belief = self.reparameterize(mu, logvar)

        return belief, hidden

    def init_hidden(self, batchsize):
        hidden = torch.zeros((self.args.gru_hidden_layers, batchsize, self.hidden_dim))
        return hidden
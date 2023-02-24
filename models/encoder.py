import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args

        # fc layer before passing into GRU units
        # TODO: add option to make multiple layers
        self.hidden_dim = args.gru_hidden_size

        self.fc_embedding = nn.Linear(args.num_features, args.fc_dim)
        
        self.fc_correct = nn.Linear(args.fc_dim, 64)
        self.fc_rest = nn.Linear((args.query_size-1) * args.fc_dim, 64)

        # RNN functionality
        self.gru = nn.GRU(input_size=128,
                          hidden_size=self.hidden_dim,
                          num_layers=args.gru_hidden_layers)

        # fc layers after the RNN
        fc_layers = []
        curr = self.hidden_dim
        for l in args.fc_layers:
            fc_layers.append(nn.Linear(curr, l).to(args.device))
            curr = l
        self.fc_layers = fc_layers

        self.fc_mu = nn.Linear(curr, args.num_features)
        self.fc_logvar = nn.Linear(curr, args.num_features)

    def forward(self, query, hidden):
        # embed all query inputs
        embeddings = self.fc_embedding(query)

        # separate first query from the rest
        correct_embedding = embeddings[:, :, 0, :].unsqueeze(-2)
        rest_embeddings = embeddings[:, :, 1:, :]
        correct_embedding = torch.flatten(correct_embedding, start_dim=-2)
        rest_embeddings = torch.flatten(rest_embeddings, start_dim=-2)

        # fc layer
        correct = self.fc_correct(correct_embedding)
        correct = F.leaky_relu(correct)
        rest = self.fc_rest(rest_embeddings)
        rest = F.leaky_relu(rest)
        output = torch.cat((correct, rest), dim=-1)

        # run through gru
        output, hidden = self.gru(output, hidden)

        # run through fc layers
        for l in self.fc_layers:
            output = l(output)
            output = F.leaky_relu(output)

        # output belief distribution and belief sample
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)

        return mu, logvar, hidden

    def init_hidden(self, batchsize):
        hidden = torch.zeros((self.args.gru_hidden_layers, batchsize, self.hidden_dim)).to(self.args.device)
        return hidden
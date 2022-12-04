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

        self.fc_input_query = nn.Linear(args.query_size * args.num_features, args.fc_dim)
        self.fc_input_answer = nn.Linear(1, args.fc_dim)

        self.fc_input = nn.Linear(2 * args.fc_dim, 4 * args.fc_dim)

        # RNN functionality
        self.gru = nn.GRU(input_size=4 * args.fc_dim,
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
            std = torch.exp(0.5*logvar).repeat(self.args.m, 1, 1)
            eps = torch.randn_like(std)
            belief = eps.mul(std).add_(mu.repeat(self.args.m, 1, 1))
            belief = torch.mean(belief, dim=0)
            belief = belief / torch.linalg.norm(belief, dim=-1).unsqueeze(-1)
        else:
            belief = mu / torch.linalg.norm(mu, dim=-1).unsqueeze(-1)
        return belief

    def forward(self, query, answer, hidden):
        input_query = self.fc_input_query(query)
        input_query = F.relu(input_query)

        input_answer = self.fc_input_answer(answer.to(torch.float32))
        input_answer = F.relu(input_answer)
        
        input = torch.cat((input_query, input_answer), -1)
        output = self.fc_input(input)
        output = F.relu(output)

        # run through gru
        output, hidden = self.gru(output, hidden)

        # run through fc layers
        for l in self.fc_layers:
            output = l(output)
            output = F.relu(output)

        # output belief distribution and belief sample
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        belief = self.reparameterize(mu, logvar)

        return belief, hidden

    def init_hidden(self, batchsize):
        hidden = torch.zeros((1, batchsize, self.hidden_dim))
        return hidden
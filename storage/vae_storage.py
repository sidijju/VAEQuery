import numpy as np
import torch
from query.simulate import response_dist, sample_dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def order_queries(queries, answers):
    ordered_queries = torch.zeros_like(queries).to(device)
    for b in range(len(queries)):
        # shuffle queries so that the chosen query is first
        idx = list(range(len(queries[0])))
        idx.insert(0, idx.pop(answers[b]))
        ordered_queries[b] = queries[b][idx]
    return ordered_queries

def respond_queries(args, queries, true_rewards):
    dists = response_dist(args, queries, true_rewards)
    answers = sample_dist(args, dists)
    return answers

class VAEStorage:

    # Storage of queries for VAE training
    
    def __init__(self, args):

        self.args = args

        assert args.num_features > 0

        # max number of queries
        self.buffer_size = args.buffer_size
        assert self.buffer_size > 0

        self.buffer_len = 0
        self.idx = 0

        self.queries = torch.zeros((self.buffer_size, args.query_size, args.num_features)).to(device)

        self.mean = None
        self.std = None

    def normalize_dataset(self, mean=None, std=None):
        if mean == None:
            mean = torch.mean(self.queries[:self.buffer_len], dim=(0,1)).to(device)
            self.mean = mean
        if std == None:
            std = torch.std(self.queries[:self.buffer_len], dim=(0,1)).to(device)
            self.std = std

        self.queries = torch.sub(self.queries, mean)
        self.queries = torch.div(self.queries, std)

    def get_random_true_rewards(self, batchsize=5):
        true_rewards = torch.randn(batchsize, self.args.num_features).to(device)
        true_rewards = true_rewards / torch.norm(true_rewards, dim=-1).unsqueeze(-1)
        return true_rewards

    def get_random_queries(self, batchsize=5):
         # select random indices for queries
        size = min(self.buffer_len, batchsize)
        idx = np.random.choice(range(self.buffer_len), size, replace=False)
        queries = self.queries[idx, :, :]
        return queries
   
    def get_batch(self, batchsize=5, true_rewards=None):
        # get true rewards if None
        if true_rewards is None:
            true_rewards = self.get_random_true_rewards(batchsize=batchsize)
        assert batchsize == len(true_rewards)

        # get queries and responses from dataset
        queries = self.get_random_queries(batchsize=batchsize)
        answers = respond_queries(self.args, queries, true_rewards)

        return true_rewards, queries, answers

    def get_batch_seq(self, batchsize=5, seqlength=10, true_rewards=None):
        # get true rewards if None
        if true_rewards is None:
            true_rewards = self.get_random_true_rewards(batchsize=batchsize)
        assert batchsize == len(true_rewards)

        # generate query and answer sequences
        query_seqs = []
        answer_seqs = []
        for _ in range(seqlength):
            # select the rollouts we want and append to sequences
            _, queries, answers = self.get_batch(batchsize=batchsize, true_rewards=true_rewards)
            query_seqs.append(queries)
            answer_seqs.append(answers)

        query_seqs = torch.stack(query_seqs).to(device)   
        answer_seqs = torch.stack(answer_seqs).to(device)

        return true_rewards, query_seqs, answer_seqs

    def insert(self, query):

        # ring buffer, replace at beginning
        if self.idx >= self.buffer_size:
            self.buffer_len = self.buffer_size
            self.idx = 0
        else:
            self.buffer_len = min(self.buffer_len + 1, self.buffer_size)

        # add to larger buffer
        self.queries[self.idx] = query
        self.idx += 1
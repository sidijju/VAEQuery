import numpy as np
import torch
import torch.nn.functional as F
from query.simulate import response_dist, sample_dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.queries = torch.zeros((self.buffer_size, args.query_size, args.num_features))

        self.mean = None
        self.std = None

    def normalize_dataset(self, mean=None, std=None):
        if mean == None:
            mean = torch.mean(self.queries[:self.buffer_len], dim=0)
            self.mean = mean
        if std == None:
            std = torch.std(self.queries[:self.buffer_len], dim=0)
            self.std = std

        for i in range(self.buffer_len):
            self.queries[i] = (self.queries[i] - mean) / std

    def get_random_true_rewards(self, batchsize=5):
        true_rewards = torch.normal(0, 1, (batchsize, self.args.num_features))
        true_rewards = true_rewards / (torch.norm(true_rewards, dim=-1).unsqueeze(-1))
        return true_rewards

    def get_random_queries(self, batchsize=5):
         # select random indices for queries
        size = min(self.buffer_len, batchsize)
        idx = np.random.choice(range(self.buffer_len), size, replace=False)
        queries = self.queries[idx, :, :]
        return queries
   
    def get_batch(self, batchsize=5, true_rewards=None):
        # get batchsize queries and responses from dataset
        queries = self.get_random_queries(batchsize=batchsize)

        # get true rewards if None
        if true_rewards is None:
            true_rewards = self.get_random_true_rewards(batchsize=batchsize)
        assert batchsize == len(true_rewards)

        dists = response_dist(self.args, queries, true_rewards)
        answers = []
        for b in range(batchsize):
            answers.append(sample_dist(self.args, dists[b]))
            # shuffle queries so that the chosen query is first
            idx = list(range(self.args.query_size))
            idx.insert(0, idx.pop(answers[b]))
            queries[b] = queries[b][idx]
        answers = torch.stack(answers)

        # select the rollouts we want
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

        query_seqs = torch.stack(query_seqs)   
        answer_seqs = torch.stack(answer_seqs)

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
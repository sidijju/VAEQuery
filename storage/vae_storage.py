import numpy as np
import torch
import torch.nn.functional as F
from query.simulate import response_dist

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

        self.queries = torch.zeros((self.buffer_size, args.query_size * args.num_features))
        
    def get_batch(self, batchsize=5):
        # get batchsize queries from dataset
        # queries consist of query_size trajectories plus the response

        size = min(self.buffer_len, batchsize)

        # select random indices for trajectories
        idx = np.random.choice(range(self.buffer_len), size, replace=False)

        true_rewards = torch.rand((batchsize, self.args.num_features))
        true_rewards = true_rewards / (torch.sum(true_rewards, 1).unsqueeze(-1))

        queries = self.queries[idx, :]

        dists = response_dist(self.args, queries, true_rewards)
        answers = []
        for b in range(batchsize):
            answers.append(torch.multinomial(dists[b], 1))
        answers = torch.stack(answers)

        assert 1 == 0

        # select the rollouts we want
        return true_rewards, queries, answers

    def get_batch_seq(self, batchsize=5, seqlength=10):
        # get batchsize sequences queries from dataset
        # queries consist of query_size trajectories plus the response

        size = min(self.buffer_len, batchsize)

        # generate true humans for each query sequence
        true_rewards = torch.rand((batchsize, self.args.num_features))
        true_rewards = true_rewards / (torch.sum(true_rewards, 1).unsqueeze(-1))

        # generate query sequences
        query_seqs = []
        for _ in range(seqlength):
            # select random indices for trajectories
            idx = np.random.choice(range(self.buffer_len), size, replace=False)

            # select the rollouts we want and append to sequences
            query_seqs.append(self.queries[idx, :])
        query_seqs = torch.stack(query_seqs)

        # generate answer sequences
        # don't require answers to have gradient so this is okay
        dists = response_dist(self.args, query_seqs, true_rewards)
        answer_seqs = []
        for t in range(seqlength):
            answer_seq = torch.multinomial(dists[t], 1)
            answer_seqs.append(answer_seq)
        answer_seqs = torch.stack(answer_seqs)

        return true_rewards, query_seqs, answer_seqs

    def insert(self, query):

        # ring buffer, replace at beginning
        if self.idx >= self.buffer_size:
            self.buffer_len = self.buffer_size
            self.idx = 0
        else:
            self.buffer_len += 1

        # add to larger buffer
        self.queries[self.idx] = query
        self.idx += 1
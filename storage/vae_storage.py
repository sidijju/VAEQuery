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

        self.queries = torch.zeros((self.buffer_size, args.query_size * args.num_features))

    def get_random_true_rewards(self, batchsize=5):
        true_rewards = torch.normal(0, 1, (batchsize, self.args.num_features))
        true_rewards = true_rewards / (torch.norm(true_rewards, 1).unsqueeze(-1))
        return true_rewards

    def get_random_queries(self, batchsize=5):
         # select random indices for queries
        size = min(self.buffer_len, batchsize)
        idx = np.random.choice(range(self.buffer_len), size, replace=False)
        queries = self.queries[idx, :]
        return queries
   
    def get_batch(self, batchsize=5):
        # get batchsize queries and responses from dataset
        queries = self.get_random_queries(batchsize=batchsize)

        # get true rewards
        true_rewards = self.get_random_true_rewards(batchsize=batchsize)

        dists = response_dist(self.args, queries, true_rewards)
        answers = []
        for b in range(batchsize):
            answers.append(sample_dist(self.args, dists[b]))
        answers = torch.stack(answers)
        
        # select the rollouts we want
        return true_rewards, queries, answers

    def get_batch_seq(self, batchsize=5, seqlength=10):
        # generate query sequences
        query_seqs = []
        for _ in range(seqlength):
            # select the rollouts we want and append to sequences
            queries = self.get_random_queries(batchsize=batchsize)
            query_seqs.append(queries)
        query_seqs = torch.stack(query_seqs)

        # get true rewards for each query sequence
        true_rewards = self.get_random_true_rewards(batchsize=batchsize)

        # generate answer sequences
        # detach rewards to prevent gradients
        dists = response_dist(self.args, query_seqs, true_rewards)
        answer_seqs = []
        for t in range(seqlength):
            answer_seq = sample_dist(self.args, dists[t])
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
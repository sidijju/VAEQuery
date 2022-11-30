import numpy as np
import torch
import torch.nn.functional as F

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
        self.answers = torch.zeros((self.buffer_size, 1))
        
    def get_batch(self, batchsize=5):
        # get batchsize queries from dataset
        # queries consist of query_size trajectories plus the response

        size = min(self.buffer_len, batchsize)

        # select random indices for trajectories
        idx = np.random.choice(range(self.buffer_len), size, replace=False)

        # select the rollouts we want
        return self.queries[idx, :], self.answers[idx]

    def get_batch_seq(self, batchsize=5, seqlength=10):
        # get batchsize sequences queries from dataset
        # queries consist of query_size trajectories plus the response

        size = min(self.buffer_len, batchsize)

        query_seqs = []
        answer_seqs = []

        for _ in range(seqlength):
            # select random indices for trajectories
            idx = np.random.choice(range(self.buffer_len), size, replace=False)

            # select the rollouts we want and append to sequences
            query_seqs.append(self.queries[idx, :])
            answers = self.answers[idx].to(int)
            answer_seqs.append(answers)

        query_seqs = torch.stack(query_seqs)
        answer_seqs = torch.stack(answer_seqs)
        answer_seqs = torch.squeeze(F.one_hot(answer_seqs, num_classes=self.args.query_size))
        answer_seqs = answer_seqs.to(float)

        return query_seqs, answer_seqs

    def insert(self, query, answer):

        # ring buffer, replace at beginning
        if self.idx >= self.buffer_size:
            self.buffer_len = self.buffer_size
            self.idx = 0
        else:
            self.buffer_len += 1

        # add to larger buffer
        self.queries[self.idx] = query
        self.answers[self.idx] = answer
        self.idx += 1
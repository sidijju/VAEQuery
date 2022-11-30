import numpy as np
import torch
import torch.nn.functional as F
from query.simulate import SimulatedHuman

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

        true_humans = []
        for _ in range(batchsize):
            true_humans.append(SimulatedHuman(self.args))

        # select the rollouts we want
        return true_humans, self.queries[idx, :]

    def get_batch_seq(self, batchsize=5, seqlength=10):
        # get batchsize sequences queries from dataset
        # queries consist of query_size trajectories plus the response

        size = min(self.buffer_len, batchsize)

        # generate true humans for each query sequence
        true_humans = []
        for _ in range(batchsize):
            true_humans.append(SimulatedHuman(self.args))

        # generate query sequences
        query_seqs = []
        for _ in range(seqlength):
            # select random indices for trajectories
            idx = np.random.choice(range(self.buffer_len), size, replace=False)

            # select the rollouts we want and append to sequences
            query_seqs.append(self.queries[idx, :])

        # generate answer sequences
        answer_seqs = []
        for i in range(batchsize):
            answer_seq = []
            true_human = true_humans[i]
            for t in range(seqlength):
                answer_seq.append(true_human.response(query_seqs[t][i]))
            answer_seqs.append(torch.tensor(answer_seq))

        query_seqs = torch.stack(query_seqs)
        answer_seqs = torch.stack(answer_seqs)
        answer_seqs = torch.movedim(answer_seqs, 1, 0)

        return true_humans, query_seqs, answer_seqs

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
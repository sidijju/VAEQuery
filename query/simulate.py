import numpy as np
import torch
from torch.nn.functional import normalize

class SimulatedHuman:
    def __init__(self, args, w=None):
        self.num_features = args.num_features
        self.dist_size = args.query_size
        self.temperature = args.temperature

        # initialize true human reward if not given
        if w is None:
            w = np.random.random(self.num_features)
            w = torch.tensor(w)
            
        self.w = normalize(w, dim=0).to(torch.float64)

    def response(self, query):
        return self.sample(self.response_dist(query))

    def response_dist(self, query):
        dist = torch.zeros(self.dist_size)
        for t in range(self.dist_size):
            start = t * self.num_features
            traj = query[start:start+self.num_features]
            dist[t] = torch.exp(1/self.temperature * torch.dot(traj.to(torch.float64), self.w))
        dist /= dist.sum()
        assert abs(dist.sum() - 1) < 1e-4
        return dist

    def sample(self, dist):
        for i in range(1, len(dist)):
            dist[i] += dist[i-1]

        rand = np.random.random_sample()
        for i in range(len(dist)):
            if rand <= dist[i]:
                return i
        return len(dist)-1

    def alignment(self, other):
        return abs(torch.dot(other.w, self.w)/(torch.linalg.norm(other.w)*torch.linalg.norm(self.w)))
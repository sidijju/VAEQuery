import numpy as np

class SimulatedHuman:
    def __init__(self, args):
        self.num_features = args.num_features
        self.dist_size = args.query_size
        self.temperature = args.temperature

        # initialize true human reward
        w = np.random.random(self.num_features)
        w /= w.sum()
        self.w = w

    def response(self, query):
        return self.sample(self.response_dist(query))

    def response_dist(self, query):
        dist = np.zeros(self.dist_size)
        for t in range(self.dist_size):
            start = t * self.num_features
            traj = query[start:start+self.num_features]
            dist[t] = np.exp(1/self.temperature * np.dot(self.w, traj))
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
import numpy as np

def simulate_human(args, w, query):
    dist = np.zeros(args.query_size)
    for t in range(args.query_size):
        start = t * args.num_features
        traj = query[start:start+args.num_features]
        dist[t] = np.exp(1/args.temperature * np.dot(w, traj))

    assert abs(dist.sum() - 1) < 1e-4

    for i in range(1, len(dist)):
        dist[i] += dist[i-1]

    rand = np.random.random_sample()
    for i in range(len(dist)):
        if rand <= dist[i]:
            return i
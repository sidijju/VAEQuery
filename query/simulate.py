import numpy as np
import torch
from torch.nn.functional import softmax

def response_dist(args, query, w):
    trajs = torch.reshape(query, (*query.shape[:-1], args.query_size, args.num_features))
    w = w.unsqueeze(-2)
    dist = torch.sum(trajs * w, dim=-1)
    dist = softmax(dist * 1/args.temperature, dim=-1)
    return dist

def alignment(w1, w2):
    return abs(torch.sum(w1 * w2, dim=-1)/(torch.linalg.norm(w1)*torch.linalg.norm(w2)))
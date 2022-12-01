import numpy as np
import torch
from torch.nn.functional import softmax

def response_dist(args, query, w):
    trajs = torch.reshape(query, (*query.shape[:-1], args.query_size, args.num_features))
    trajs = torch.transpose(trajs, -1, -2)
    w = w.unsqueeze(-1)
    dist = torch.sum(trajs * w, dim=-2)
    dist *= 1/args.temperature
    return softmax(dist, dim=-1)

def alignment(w1, w2):
    return abs(torch.sum(w1 * w2, dim=-1)/(torch.linalg.norm(w1)*torch.linalg.norm(w2)))
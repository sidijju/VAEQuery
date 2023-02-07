import numpy as np
import torch
from torch.nn.functional import softmax

def sample_dist(args, dist):
    probs = softmax(dist, dim=-1)
    if args.optimal_user:
        sample = torch.argmax(dist, dim=-1).unsqueeze(-1)
    else:
        sample = torch.multinomial(probs, 1)
    return sample

def response_dist(args, query, w):
    return torch.bmm(query, w.unsqueeze(-1)).squeeze(-1) * args.rationality

def alignment(w1, w2):
    return abs(torch.sum(w1 * w2, dim=-1)/(torch.linalg.norm(w1, dim=-1)*torch.linalg.norm(w2, dim=-1)))
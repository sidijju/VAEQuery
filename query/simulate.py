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
    w = w.unsqueeze(-2)
    dist = torch.sum(query * w, dim=-1)
    # boltzmann rationality
    dist *= 1/args.temperature
    # return logits, used in sample_dist later (softmax applied then)
    return dist

def alignment(w1, w2):
    alignments = abs(torch.sum(w1 * w2, dim=-1)/(torch.linalg.norm(w1, dim=-1)*torch.linalg.norm(w2, dim=-1)))
    return alignments
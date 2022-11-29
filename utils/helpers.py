import os
import numpy as np
import torch
from visualize import visualize_behavior
from storage.vae_storage import VAEStorage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makedir(dirname = "visualizations"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def sample_gaussian(mu, logvar):
    stddev = torch.exp(0.5 * logvar)
    norm_sample = torch.randn_like(stddev)
    return norm_sample.mul(stddev).add(mu)

class FeatureExtractor:

    #return features from a trajectory

    #update driver.py with num_features if change in number of features

    def __init__(self):
        pass

    def featurize(self, traj):
        features = 5
        obs, actions, next_obs, rews = traj
        pass

def collect_random_trajectory(world):
    world.reset()

    # horizon number of (s, a, s', r) tuples
    obs = np.zeros((world.horizon, world.state_dim))
    actions = np.zeros((world.horizon, world.action_dim))
    next_obs = np.zeros((world.horizon, world.state_dim))
    rews = np.zeros((world.horizon, world.state_dim))

    for i in range(world.horizon):
        obs[i] = world.state
        action = world.action_space.sample()
        actions[i] = action
        next_ob, rew, _, _ = world.step(action)
        next_obs[i] = next_ob
        rews[i] = rew

    return obs, actions, next_obs, rews

def collect_dataset(args, world, human):
    feature_extractor = FeatureExtractor(args)
    dataset = VAEStorage(args)

    assert args.precollect_num > args.batchsize

    for _ in range(args.precollect_num):
        query = []
        for _ in range(args.query_size):
            traj = collect_random_trajectory(world)
            featurized = feature_extractor.featurize(traj)
            query.extend(featurized)
        
        answer = human.response(query)
        dataset.insert(query, answer)
    
    return dataset
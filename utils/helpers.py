import os
import numpy as np
import torch
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

    def __init__(self, args):
        self.args = args
        pass

    def featurize(self, traj):
        features = np.zeros((self.args.num_features))
        obs, actions, next_obs, rews = traj
        
        if self.args.env_type == 'gridworld':
            #feature 1 - total reward
            features[0] = rews.sum()

            #feature 2 - number of actions that aren't stay
            features[1] = len(actions) - np.count_nonzero(actions == 4)

            #feature 3 - 

            #feature 4 - 

            #feature 5 - 
        
        return features


def collect_random_trajectory(world):
    world.reset()

    # horizon number of (s, a, s', r) tuples
    obs = torch.zeros((world.horizon, world.state_dim))
    actions = torch.zeros((world.horizon, world.action_dim))
    next_obs = torch.zeros((world.horizon, world.state_dim))
    rews = torch.zeros((world.horizon, 1))

    for i in range(world.horizon):
        obs[i] = torch.tensor(world.state)
        action = world.action_space.sample()
        actions[i] = torch.tensor(action)
        next_ob, rew, _, _ = world.step(action)
        next_obs[i] = next_ob
        rews[i] = rew

    return obs, actions, next_obs, rews

def collect_dataset(args, world):
    feature_extractor = FeatureExtractor(args)
    dataset = VAEStorage(args)

    assert args.precollect_num > args.batchsize

    if args.verbose:
        print("########### COLLECTING DATASET ###########")

    for i in range(args.precollect_num):
        query = []
        for _ in range(args.query_size):
            traj = collect_random_trajectory(world)
            featurized = feature_extractor.featurize(traj)
            query.extend(featurized)
        
        assert len(query) == args.query_size * args.num_features
        
        query = torch.tensor(query)
        dataset.insert(query)

        if args.verbose and i % 1000 == 0:
            print("Collected %2d queries" % (i))
    
    return dataset
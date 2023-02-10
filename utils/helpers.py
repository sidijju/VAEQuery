import os
import numpy as np
import torch
from storage.vae_storage import VAEStorage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def makedir(dirname = "visualizations"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

class FeatureExtractor:

    #return features from a trajectory

    #update driver.py with num_features if change in number of features

    def __init__(self, args):
        self.args = args

    def featurize(self, traj):
        features = torch.zeros((self.args.num_features))
        obs, actions, next_obs, rews = traj
        
        if self.args.env_type == 'gridworld':
            # #feature 0 - total reward
            # features[0] = rews.sum()

            # #feature 1 - number of actions that aren't stay
            # features[1] = len(actions) - np.count_nonzero(actions == 4)

            # #feature 2 - average distance from bottom left
            # features[2] = sum([distance(ob[0], 0, ob[1], 0) for ob in next_obs])/len(obs)

            # #feature 3 - average distance from top right
            # features[3] = sum([distance(ob[0], self.args.grid_size, ob[1], self.args.grid_size) for ob in next_obs])/len(obs)

            # #feature 4 - average distance from bottom right
            # features[4] = sum([distance(ob[0], self.args.grid_size, ob[1], 0) for ob in next_obs])/len(obs)

            # #feature 5 - average distance from top left
            # features[5] = sum([distance(ob[0], 0, ob[1], self.args.grid_size) for ob in next_obs])/len(obs)

            ### ONLY FOUR FEATURES ###

            #feature 0 - total reward
            features[0] = rews.sum()

            #feature 1 - number of actions that aren't stay
            features[1] = len(actions) - np.count_nonzero(actions == 4)

            #feature 2 - average x distance
            features[2] = sum([distance(ob[0], 0, 0, 0) for ob in next_obs])/len(obs)

            #feature 3 - average y distance
            features[3] = sum([distance(0, 0, ob[1], 0) for ob in next_obs])/len(obs)

        #return features
        return torch.randn(self.args.num_features)

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

def collect_dataset(args, world, mean=None, std=None):
    feature_extractor = FeatureExtractor(args)
    dataset = VAEStorage(args)

    assert args.precollect_num >= args.batchsize

    if args.verbose:
        print("########### COLLECTING DATASET ###########")

    for i in range(args.precollect_num):
        query = []
        for _ in range(args.query_size):
            traj = collect_random_trajectory(world)
            featurized = feature_extractor.featurize(traj)
            query.append(featurized)
        
        query = torch.stack(query)
        dataset.insert(query)

        if args.verbose and (i+1) % 200 == 0:
            print("Collected %2d queries" % (i+1))
    
    dataset.normalize_dataset(mean=mean, std=std)
    return dataset

def reparameterize(args, mu, logvar, samples=1):
    if args.sample_belief:
        belief = (torch.randn(samples, args.batchsize, args.num_features) * torch.sqrt(torch.exp(logvar))) + mu
        belief = belief.transpose(0, 1)
    else:
        belief = mu.unsqueeze(1).repeat(1, samples, 1)
    return belief
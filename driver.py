from environments.envs import *
from configs import gridworld
from utils.helpers import makedir, collect_dataset
from policies.policies import RandomPolicy, GreedyPolicy, RLPolicy
from learner import Learner
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', default='gridworld')
parser.add_argument('--exp_name', default='')
args, rargs = parser.parse_known_args()
env = args.env_type
exp_name = args.exp_name

#### set up argument variables ####
if env == 'gridworld':
    args = gridworld.args(rargs)
    args.num_features = 6
    args.env_type = env
    args.exp_name = exp_name

    # collect dataset for the experiments
    dataset = collect_dataset(args, GridWorld(args))

else:
    print('Invalid Environment Option')

#### create directories for visualizations ####
if args.visualize:
        makedir("visualizations")  
        makedir("visualizations/" + args.exp_name)

#### set up learners for each policy ####

rand_learner = Learner(args, dataset, RandomPolicy)
#greedy_learner = Learner(args, dataset, GreedyPolicy) TODO
#rl_learner = Learner(args, dataset, RLPolicy) TODO

#### run training for each policy ####

# run training for random policy
rand_learner.train()

# run training for greedy policy TODO
# greedy_learner.train()

# run training for rl policy TODO
# rl_learner.train()
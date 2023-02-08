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
    #args.num_features = 6
    args.num_features = 4
    args.env_type = env
    args.exp_name = exp_name

    # collect datasets for the experiments
    train_dataset = collect_dataset(args, GridWorld(args))
    test_dataset = collect_dataset(args, GridWorld(args), mean=train_dataset.mean, std=train_dataset.std)
    datasets = (train_dataset, test_dataset)

else:
    print('Invalid Environment Option')

#### create directories for runs ####
makedir("logs")
makedir("logs/" + args.exp_name)
args.log_dir = "logs/" + args.exp_name + "/"

#### set up learners for each policy ####

rand_learner = Learner(args, datasets, RandomPolicy)
greedy_learner = Learner(args, datasets, GreedyPolicy)
#rl_learner = Learner(args, dataset, RLPolicy) TODO

#### run training for each policy ####

# run training and testing for random policy
rand_learner.train()
rand_learner.test()

# run training for greedy policy TODO
greedy_learner.train()
greedy_learner.test()

# run training for rl policy TODO
# rl_learner.train()
# rl_learner.test()
from environments.envs import *
from visualize import visualize
from configs import gridworld
from policies.policies import RandomPolicy, GreedyPolicy, RLPolicy
from learner import Learner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env-type', default='gridworld')
args, rargs = parser.parse_known_args()
env = args.env_type

if env == 'gridworld':
    args = gridworld.args(rargs)
    args.num_features = 5
    args.env_type == env
    
    # run training for random policy
    rand_learner = Learner(args, GridWorld(args), RandomPolicy())

    # run training for greedy policy
    # greedy_learner = Learner(args, GridWorld(args), GreedyPolicy())

    # run training for rl policy
    # rl_learner = Learner(args, GridWorld(args), RLPolicy())

else:
    print('Invalid Environment Option')
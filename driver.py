from environments.envs import *
from configs import gridworld
from utils.helpers import makedir
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

if env == 'gridworld':
    args = gridworld.args(rargs)
    args.num_features = 5
    args.env_type = env
    args.exp_name = str(time.time()) + "-" + exp_name

    if args.visualize:
        makedir("visualizations")  
        makedir("visualizations/" + args.exp_name)

    rand_learner = Learner(args, GridWorld(args), RandomPolicy(), exp_name=args.exp_name)
    #greedy_learner = Learner(args, GridWorld(args), GreedyPolicy(), exp_name=args.exp_name)
    #rl_learner = Learner(args, GridWorld(args), RLPolicy(),  exp_name=args.exp_name)

    # run training for random policy
    rand_learner.train()

    # run training for greedy policy
    # greedy_learner.train()

    # run training for rl policy
    # rl_learner.train()

else:
    print('Invalid Environment Option')
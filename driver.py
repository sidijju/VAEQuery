from environments.envs import *
from configs import gridworld
from policies.policies import RandomPolicy, GreedyPolicy, RLPolicy
from learner import Learner
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', default='gridworld')
args, rargs = parser.parse_known_args()
env = args.env_type

if env == 'gridworld':
    args = gridworld.args(rargs)
    args.num_features = 5
    args.env_type = env

    timestr = str(time.time())
    
    # run training for random policy
    rand_learner = Learner(args, GridWorld(args), RandomPolicy(), exp_name=timestr)

    # run training for greedy policy
    # greedy_learner = Learner(args, GridWorld(args), GreedyPolicy(), exp_name=timestr)

    # run training for rl policy
    # rl_learner = Learner(args, GridWorld(args), RLPolicy(),  exp_name=timestr)

else:
    print('Invalid Environment Option')
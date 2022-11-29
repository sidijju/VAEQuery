from environments.envs import *
from visualize import visualize
from configs import gridworld
from policies import rand, greedy, rl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env-type', default='gridworld')
args, rargs = parser.parse_known_args()
env = args.env_type

if env == 'gridworld':
    args = gridworld.args(rargs)
    args.state_dim = 2
    args.action_dim = 1
    args.task_dim = 2
    args.env_type == env
    # visualize learning of policies
    visualize(GridWorld(args), rand.run_policy, fname="random")
    visualize(GridWorld(args), greedy.run_policy, fname="greedy")
    visualize(GridWorld(args), rl.run_policy, fname="rl")
else:
    print('Invalid Environment Option')
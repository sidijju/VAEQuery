from environments.envs import *
from visualize import visualize
from models import *
from configs import gridworld
from policies import rand, greedy, rl
from utils.helpers import collect_trajectories
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env-type', default='gridworld')
args, rargs = parser.parse_known_args()
env = args.env_type

if env == 'gridworld':
    args = gridworld.args(rargs)
    args.num_features = 5
    args.env_type == env
    
    # collect dataset on world
    world = GridWorld(args)
    dataset = collect_trajectories(world, n=args.precollect_num)

else:
    print('Invalid Environment Option')

def train(args, policy, dataset):
    # initialize true human reward
    w = np.random.random(args.num_features)
    w /= w.sum()

    # initialize encoder network
    encoder = encoder.Encoder(args)

    # initialize belief network
    belief = belief.Belief(args)
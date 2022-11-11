from configs import gridworld
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
    # visualize learning of random policy
    visualize(GridWorld(), policies.random_policy.run_policy, fname="random")
    # visualize learning of varibadPolicy
    #visualize(env, MetaLearner(args), fname="varibad")
else:
    print('Invalid Environment Option')
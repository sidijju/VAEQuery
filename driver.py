from environments.envs import *
from configs import gridworld
from utils.helpers import makedir, collect_dataset, set_seed
from policies.policies import *
from learner import Learner
import argparse
import matplotlib.pyplot as plt

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
    #args.num_features = 4
    args.env_type = env
    args.exp_name = exp_name
    args.device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    # set up random seed
    set_seed(args.seed)

    # collect datasets for the experiments
    train_dataset = collect_dataset(args, GridWorld(args))
    test_dataset = collect_dataset(args, GridWorld(args), mean=train_dataset.mean, std=train_dataset.std)
    datasets = (train_dataset, test_dataset)

else:
    # TODO add more environments
    print('Invalid Environment Option')

#### create directories for runs ####
makedir("logs")
makedir("logs/" + args.exp_name)
args.log_dir = "logs/" + args.exp_name + "/"

#### set up learners for each policy ####

rand_learner = Learner(args, datasets, RandomPolicy)

if args.random_encoder:
    greedy_learner = Learner(args, datasets, GreedyPolicy, encoder=rand_learner.encoder)
    rl_learner = Learner(args, datasets, RLStatePolicy, encoder=rand_learner.encoder)
    rl_feed_learner = Learner(args, datasets, RLFeedPolicy, encoder=rand_learner.encoder)
else:
    greedy_learner = Learner(args, datasets, GreedyPolicy)
    rl_learner = Learner(args, datasets, RLStatePolicy)
    rl_feed_learner = Learner(args, datasets, RLFeedPolicy)

# set up storage for results
labels = []
train_losses = []
mses = []
alignments = []

#### run training for each policy ####

learners = [rand_learner, greedy_learner, rl_learner, rl_feed_learner]

# get shared test batch for all learners
test_batch = test_dataset.get_batch(batchsize=args.batchsize)

for learner in learners:
    # run training and testing for policy
    train_losses.append(learner.train())
    mse, align = learner.test(batch=test_batch)
    mses.append(mse)
    alignments.append(align)
    labels.append(learner.policy.vis_directory[:-1])

if args.visualize:
    for i in range(len(train_losses)):
        plt.plot(train_losses[i], label=labels[i])
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Loss over Training")
    plt.savefig(args.log_dir + "losses")
    plt.close()

    for i in range(len(train_losses)):
        plt.plot(mses[i], label=labels[i])
    plt.legend()
    plt.xlabel("Queries")
    plt.xticks(range(1, args.sequence_length+1))
    plt.ylabel("MSE")
    plt.title("Reward Error")
    plt.savefig(args.log_dir + "errors")
    plt.close()

    for i in range(len(train_losses)):
        plt.plot(alignments[i], label=labels[i])
    plt.legend()
    plt.xlabel("Queries")
    plt.xticks(range(1, args.sequence_length+1))
    plt.ylabel("Alignment")
    plt.title("Reward Alignment")
    plt.savefig(args.log_dir + "alignments")
    plt.close()

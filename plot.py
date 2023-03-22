import torch
from models.encoder import *
import matplotlib.pyplot as plt
from utils.helpers import collect_dataset, reparameterize, makedir, set_seed
from configs import gridworld
import argparse
from environments.envs import *
from storage.vae_storage import order_queries, respond_queries
from query.simulate import response_dist, alignment
from stable_baselines3 import A2C

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', default='gridworld')
parser.add_argument('--exp_name', default='')
args, rargs = parser.parse_known_args()
env = args.env_type
exp_name = args.exp_name
args = gridworld.args(rargs)
args.num_features = 6
args.device = torch.device("cpu")
args.env_type = "gridworld"

# set up random seed
set_seed(args.seed)

# features are random, so it should be fine that we don't normalize the dataset with respect to
# the train dataset's mean and std
dataset = collect_dataset(args, GridWorld(args))

# collect test batch
orig_true_humans, orig_queries, orig_answers = dataset.get_batch(batchsize=args.batchsize)
orig_queries = order_queries(orig_queries, orig_answers)

loss = nn.CrossEntropyLoss(reduction = 'sum')
mse_loss = nn.MSELoss()

def random_policy(mus, logvars, policy_path):
    return dataset.get_random_queries(args.batchsize)

def greedy_policy(mus, logvars, policy_path):
    queries = dataset.queries[:dataset.buffer_len]
    samples = reparameterize(args, mus, logvars, samples=args.m)
    rews = torch.einsum('qij, bmj -> bqim', queries, samples).to(args.device)
    rews = torch.exp(rews).to(args.device)
    denoms = torch.sum(rews, dim=-2).unsqueeze(-2).to(args.device)
    posteriors = rews/denoms
    mutual_answers = []
    for answer in range(args.query_size):
        sample_total = torch.sum(posteriors[:, :, answer, :], dim=-1).unsqueeze(-1)
        logmean = torch.log2(args.m * posteriors[:, :, answer, :] / sample_total)
        mutual = torch.sum(posteriors[:, :, answer, :] * logmean, dim=-1)
        mutual_answers.append(mutual)
    query_vals = -1.0/args.m * torch.sum(torch.stack(mutual_answers).to(args.device), dim=0)
    queries_chosen = queries[torch.argmin(query_vals, dim=-1)]
    return queries_chosen

def reward_function(query, answer, mus, logvars):
    rewards = 0
    for i in range(len(query)):
        q = query[i].unsqueeze(0)
        a = answer[i]
        m = mus[:, i]
        l = logvars[:, i]
        samples = reparameterize(args, m, l, samples=args.m)
        rew = torch.exp(torch.bmm(q, samples.mT))
        denom = torch.sum(rew, dim=-2).unsqueeze(-2)
        posterior = (rew/denom)[:, a]
        posterior_sum = torch.sum(posterior, dim=-1)
        
        dist = torch.distributions.MultivariateNormal(m.squeeze(), torch.diag(torch.exp(l).squeeze()))
        logprobs = dist.log_prob(samples.squeeze())
        lognum = (logprobs + torch.log2(posterior.squeeze(1)) - torch.log2(posterior_sum/args.m))
        reward = torch.sum(posterior.squeeze(1) * lognum, dim=-1)/posterior_sum - torch.sum(logprobs, dim=-1)/args.m
        rewards += reward.item()
    return rewards

def rl_policy(mus, logvars, policy_path):
    policy = A2C.load(policy_path)
    obs = torch.cat((mus, logvars), dim=-1).detach().numpy()
    queries = []
    for b in range(mus.shape[0]):
        action, _ = policy.predict(obs[b])
        queries.append(dataset.queries[action].squeeze(0))
    return torch.stack(queries).squeeze(0)

def rl_feed_policy(mus, logvars, policy_path):
    policy = A2C.load(policy_path)
    queries = []
    mus = mus.squeeze(0)
    logvars = logvars.squeeze(0)
    for b in range(mus.shape[0]):
        query = dataset.get_random_queries(batchsize=1).squeeze(0).flatten()
        obs = torch.cat((mus[b], logvars[b], query), dim=-1).detach().numpy()
        action, _ = policy.predict(obs)
        while action != 1:
            query = dataset.get_random_queries(batchsize=1).squeeze(0).flatten()
            obs[2*args.num_features:] = query.clone()
            action, _ = policy.predict(obs)
        queries.append(query.reshape(args.query_size,-1))
    return torch.stack(queries).squeeze(0)

def test(model, policy, dir, policy_path=None):
    makedir(dir)
    # set the encoder to eval mode
    model.eval()
        
    test_losses = []
    mses_mean = []
    alignments_mean = []
    cumul_rewards = []

    queries = orig_queries.clone().detach()
    answers = orig_answers.clone().detach()

    with torch.no_grad():
        hidden = model.init_hidden(batchsize=args.batchsize)
        print("#######################################################")
        for t in range(args.sequence_length):
            # get beliefs from queries and answers
            mu, logvar, hidden = model(queries.clone().unsqueeze(0), hidden)

            # get inputs and targets for cross entropy loss
            sample = reparameterize(args, mu.clone(), logvar.clone())
            sample = sample.squeeze(1)
            inputs = response_dist(args, queries, sample)
            inputs = inputs.view(-1, args.query_size)
            targets = answers.view(-1)

            # compute metrics and store in lists
            test_loss = loss(inputs, targets) / args.batchsize
            mse = mse_loss(sample, orig_true_humans)
            align = alignment(sample, orig_true_humans)

            test_losses.append(test_loss.item())
            mses_mean.append(mse.mean())
            alignments_mean.append(align.mean())

            # get next queries
            queries = policy(mu.clone(), logvar.clone(), policy_path)
            # get next answers (in the case of an actual human, would be replaced with labeling step)
            answers = respond_queries(args, queries, orig_true_humans)
            # get reward
            reward = reward_function(queries, answers, mu, logvar)
            cumul_rewards.append(reward)
            # reorder queries
            queries = order_queries(queries, answers)

            print("Query %2d: Loss = %5.3f, MSE = %5.3f, Alignment = %5.3f" % (t, test_loss, mses_mean[-1], alignments_mean[-1]))
        print("#######################################################")
    plt.plot(test_losses)
    plt.xlabel("Queries")
    plt.ylabel("CE Loss")
    plt.title("Test Evaluation - Loss")
    plt.savefig(dir + "test-loss")
    plt.close()
    plt.plot(range(1, len(mses_mean)+1), mses_mean)
    plt.xlabel("Queries")
    plt.xticks(range(1, args.sequence_length+1))
    plt.ylabel("MSE")
    plt.title("Test Evaluation - Reward Error")
    plt.savefig(dir + "test-error")
    plt.close()
    plt.plot(range(1, len(alignments_mean)+1), alignments_mean)
    plt.xlabel("Queries")
    plt.xticks(range(1, args.sequence_length+1))
    plt.ylabel("Alignment")
    plt.title("Test Evaluation - Reward Alignment")
    plt.savefig(dir + "test-alignment")
    plt.close()
    plt.plot(range(1, len(cumul_rewards)+1), cumul_rewards)
    plt.xlabel("Queries")
    plt.xticks(range(1, args.sequence_length+1))
    plt.ylabel("Cumulative Reward")
    plt.title("Test Evaluation - Cumulative Reward")
    plt.savefig(dir + "test-cumul-reward")
    plt.close()

    # return test results
    return test_losses, mses_mean, alignments_mean

makedir('crossenc')

# greedy_model = Encoder(args)
# greedy_model.load_state_dict(torch.load('logs/convergence/greedy/model.pt'))
# l1, m1, a1 = test(greedy_model, random_policy, "crossenc/greedyencrandpolicy/")

# random_model = Encoder(args)
# random_model.load_state_dict(torch.load('logs/convergence/random/model.pt'))
# l2, m2, a2 = test(random_model, random_policy, "crossenc/randencrandpolicy/")

# greedy_model = Encoder(args)
# greedy_model.load_state_dict(torch.load('logs/convergence/greedy/model.pt'))
# l3, m3, a3 = test(greedy_model, greedy_policy, "crossenc/greedyencgreedypolicy/")

# random_model = Encoder(args)
# random_model.load_state_dict(torch.load('logs/convergence/random/model.pt'))
# l4, m4, a4 = test(random_model, greedy_policy, "crossenc/randencgreedypolicy/")

# plt.plot(l1, label="greedyenc_randpolicy")
# plt.plot(l2, label="randenc_randpolicy")
# plt.plot(l3, label="greedyenc_greedypolicy")
# plt.plot(l4, label="randenc_greedypolicy")
# plt.xlabel("Queries")
# plt.xticks(range(args.sequence_length))
# plt.ylabel("CE Loss")
# plt.title("Test Evaluation - Losses")
# plt.legend()
# plt.savefig("crossenc/losses")
# plt.close()

# plt.plot(m1, label="greedyenc_randpolicy")
# plt.plot(m2, label="randenc_randpolicy")
# plt.plot(m3, label="greedyenc_greedypolicy")
# plt.plot(m4, label="randenc_greedypolicy")
# plt.xlabel("Queries")
# plt.xticks(range(args.sequence_length))
# plt.ylabel("MSE")
# plt.title("Test Evaluation - MSE")
# plt.legend()
# plt.savefig("crossenc/mse")
# plt.close()

# plt.plot(a1, label="greedyenc_randpolicy")
# plt.plot(a2, label="randenc_randpolicy")
# plt.plot(a3, label="greedyenc_greedypolicy")
# plt.plot(a4, label="randenc_greedypolicy")
# plt.xlabel("Queries")
# plt.xticks(range(args.sequence_length))
# plt.ylabel("Alignment")
# plt.title("Test Evaluation - Alignment")
# plt.legend()
# plt.savefig("crossenc/align")
# plt.close()

# plot cumulative rewards over training for RL algorithms

makedir('inc_spi')
rl_model = Encoder(args)
rl_model.load_state_dict(torch.load('logs/inc_spi/rl/model.pt'))
rll1, rlm1, rla1 = test(rl_model, rl_policy, "inc_spi/rlencrlpolicy/", 'logs/inc_spi/rl/model')

rl_feed_model = Encoder(args)
rl_feed_model.load_state_dict(torch.load('logs/inc_spi/rl-feed/model.pt'))
rlfeedl1, rlfeedm1, rlfeeda1 = test(rl_feed_model, rl_feed_policy, "inc_spi/rlfeedencrlfeedpolicy/", 'logs/inc_spi/rl-feed/model')

# plot performance of greedy model start

# makedir('hs')
# rl_model = Encoder(args)
# rl_model.load_state_dict(torch.load('logs/hot_start/rl/model.pt'))
# rll1, rlm1, rla1 = test(rl_model, rl_policy, "hs/start/", 'logs/hot_start/rl-feed/start_model.zip')
# rll2, rlm2, rla2 = test(rl_model, rl_policy, "hs/end/", 'logs/hot_start/rl-feed/model.zip')

# rl_feed_model = Encoder(args)
# rl_feed_model.load_state_dict(torch.load('logs/hot_start/rl-feed/model.pt'))
# rlfeedl1, rlfeedm1, rlfeeda1 = test(rl_feed_model, rl_feed_policy, "hs/start/", 'logs/hot_start/rl-feed/start_model')
# rlfeedl2, rlfeedm2, rlfeeda2 = test(rl_feed_model, rl_feed_policy, "hs/end/", 'logs/hot_start/rl-feed/model')


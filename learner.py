from utils.helpers import collect_trajectories, collect_dataset, sample_gaussian
from models import encoder, belief
from query.simulate import SimulatedHuman
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Learner:

    def __init__(self, args, world, policy):
        self.args = args
        self.policy = policy

        # initialize simulated true human
        self.human = SimulatedHuman(args)

        # get dataset
        trajectories = collect_trajectories(world, n=args.precollect_num)
        self.dataset = collect_dataset(trajectories, self.human)

         # initialize encoder network
        self.encoder = encoder.Encoder(args)

        # initialize belief network
        self.belief = belief.Belief(args)

        # initialize optimizer for belief and encoder networks
        params = list(self.encoder.parameters()) + list(self.belief.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr, momentum=0.9)

        # define loss function for VAE
        self.loss = nn.CrossEntropyLoss()

    def train(self, visualize=False):
        
        errors = []
        losses = []

        # pre-train encoder with untrained policy
        print("######### PRE TRAINING #########")
        for i in range(self.args.pretrain_len):
            queries, answers = self.dataset.get_batch(batchsize=self.args.batchsize)
            targets = [F.one_hot(a, num_classes=self.args.query_size) for a in answers]

            self.optimizer.zero_grad()

            latents = self.encoder(queries)
            mus, logvars = self.belief(latents)
            beliefs = [sample_gaussian(mus[i], logvars[i]) for i in range(len(mus))]
            inputs = [SimulatedHuman(beliefs[i]).response_dist(queries[i]) for i in range(len(beliefs))]

            loss = self.loss(inputs, targets)
            assert loss.shape == ()
            loss.backward()
            self.optimizer.step()

            mse = nn.MSELoss(beliefs, [self.human for _ in range(self.args.batchsize)])

            print("Iteration %2d: Loss = %.3f, MSE = %.3f" % (i, loss, mse))

            errors.append(mse)
            losses.append(loss)

        # save plots for errors and losses after pre training
        if visualize:
            plt.plot(errors)
            plt.xlabel("Iterations")
            plt.ylabel("MSE")
            plt.title("Belief vs. True Reward Error")
            plt.savefig(self.policy.vis_directory + "error")

            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.policy.vis_directory + "loss")
            
        print("########### TRAINING ###########")
        # alternate between training encoder and training policy
        # will also have to modify way we input to encoder to
        # take advantage of RNN structure
        # TODO
       

        
from utils.helpers import collect_dataset, sample_gaussian, makedir
from models import encoder, belief
from query.simulate import SimulatedHuman
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Learner:

    def __init__(self, args, world, policy, exp_name="default"):
        self.args = args
        self.policy = policy
        self.exp_name = exp_name

        # initialize simulated true human
        self.human = SimulatedHuman(args)

        # get dataset
        self.dataset = collect_dataset(args, world, self.human)

         # initialize encoder network
        self.encoder = encoder.Encoder(args)

        # initialize belief network
        self.belief = belief.Belief(args)

        # initialize optimizer for belief and encoder networks
        params = list(self.encoder.parameters()) + list(self.belief.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr, momentum=0.9)

        # define loss function for VAE
        self.loss = nn.CrossEntropyLoss()

        # directory for plots
        self.dir = self.exp_name + "/" + self.policy.vis_directory

    def train(self, visualize=False):
        
        alignments = []
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

            mse = nn.MSELoss(beliefs, [self.human.w for _ in range(self.args.batchsize)])
            alignment = [np.dot(b, self.human.w)/(np.linalg.norm(b)*np.linalg.norm(self.human.w)) for b in beliefs].mean()

            print("Iteration %2d: Loss = %.3f, MSE = %.3f, Alignment = %.3f" % (i, loss, mse, alignment))

            errors.append(mse)
            losses.append(loss)
            alignments.append(alignment)

        # save plots for errors and losses after pre training
        if visualize:
            makedir(dirname=self.dir)

            plt.plot(errors)
            plt.xlabel("Iterations")
            plt.ylabel("MSE")
            plt.title("Belief vs. True Reward Error")
            plt.savefig(self.dir + "error")

            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "loss")

            plt.plot(alignments)
            plt.xlabel("Iterations")
            plt.ylabel("Alignment")
            plt.title("Reward Alignment")
            plt.savefig(self.dir + "alignment")
            
        print("########### TRAINING ###########")
        # alternate between training encoder and training policy
        # will also have to modify way we input to encoder to
        # take advantage of RNN structure
        # TODO
       

        
from utils.helpers import collect_dataset, sample_gaussian, makedir
from models import encoder, belief
from query.simulate import SimulatedHuman
from torch import optim, nn
import torch
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
        self.optimizer = optim.Adam(params, lr=args.lr)

        # define loss function for VAE
        self.loss = nn.CrossEntropyLoss()

        # directory for plots
        self.dir = self.exp_name + "/" + self.policy.vis_directory

    def train(self, visualize=False):
        
        alignments = []
        errors = []
        losses = []

        # pre-train encoder with untrained policy
        # only train encoder - do not use policy
        print("######### PRE TRAINING #########")
        for i in range(self.args.pretrain_len):
            query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            self.optimizer.zero_grad()

            latents = self.encoder(query_seqs)

            # latents should have the output at every timestep for the input sequence
            assert latents.shape == (self.args.sequence_length, self.args.batchsize, self.args.latent_dim)

            # pass latent from last timestep into belief network
            mus, logvars = self.belief(latents[-1, :, :])

            assert mus.shape == (self.args.batchsize, self.args.num_features)
            assert logvars.shape == (self.args.batchsize, self.args.num_features)

            beliefs = [sample_gaussian(mu, logvar) for mu, logvar in zip(mus, logvars)]
            beliefs = torch.stack(beliefs)

            assert beliefs.shape == (self.args.batchsize, self.args.num_features)

            # compute inputs as the SimulatedHuman response to the query at each timestep
            humans = [SimulatedHuman(self.args, w=belief) for belief in beliefs]
            inputs = None

            assert inputs.shape == (self.args.sequence_length, self.args.batchsize, self.args.query_size)

            loss = self.loss(inputs, answer_seqs)
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
       

        
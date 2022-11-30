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

        # get dataset
        self.dataset = collect_dataset(args, world)

         # initialize encoder network
        self.encoder = encoder.Encoder(args)

        # initialize belief network
        self.belief = belief.Belief(args)

        # initialize optimizer for belief and encoder networks
        params = list(self.encoder.parameters()) + list(self.belief.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr)

        # define loss functions for VAE
        self.loss = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        # directory for plots
        if self.args.visualize:
            self.dir = "visualizations/" + self.exp_name + "/" + self.policy.vis_directory
            makedir(dirname=self.dir)
            
    def train(self):
        # pre-train encoder with untrained policy
        # only train encoder - do not use policy
        if self.args.verbose:
            print("######### PRE TRAINING #########")
            
        losses = []
        for i in range(self.args.pretrain_len):
            true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            self.optimizer.zero_grad()

            self.encoder.init_hidden()
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

            # compute inputs as the SimulatedHuman response to the query at the last timestep
            sims = [SimulatedHuman(self.args, w=belief) for belief in beliefs]

            inputs = torch.zeros((self.args.sequence_length-1, self.args.batchsize, self.args.query_size))
            for t in range(self.args.sequence_length-1):
                for b in range(self.args.batchsize):
                    inputs[t][b] = sims[t].response_dist(query_seqs[t+1][b])

            inputs = inputs.flatten(end_dim=1)
            targets = answer_seqs[1:, :].flatten(end_dim=1).squeeze()

            loss = self.loss(inputs, targets)
            assert loss.shape == ()

            loss.backward()
            self.optimizer.step()

            if self.args.verbose and i % 100 == 0:
                print("Iteration %2d: Loss = %.3f" % (i, loss))

            losses.append(loss.item())

        # save plots for errors and losses after pre training
        if self.args.visualize:
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "loss")
            plt.close()

        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### PRE TRAINING - EVALUATION #########")

        with torch.no_grad():
            true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            self.encoder.init_hidden()
            latents = self.encoder(query_seqs)

            # latents should have the output at every timestep for the input sequence
            assert latents.shape == (self.args.sequence_length, self.args.batchsize, self.args.latent_dim)

            # pass latents into belief network
            mus, logvars = self.belief(latents)

            assert mus.shape == (self.args.sequence_length, self.args.batchsize, self.args.num_features)
            assert logvars.shape == (self.args.sequence_length, self.args.batchsize, self.args.num_features)

            beliefs = []
            for t in range(self.args.sequence_length):
                beliefs_t = []
                for b in range(self.args.batchsize):
                    beliefs_t.append(sample_gaussian(mus[t][b], logvars[t][b]))
                beliefs.append(torch.stack(beliefs_t))
            beliefs = torch.stack(beliefs)

            assert beliefs.shape == (self.args.sequence_length, self.args.batchsize, self.args.num_features)

            mses = []
            alignments = []

            true_ws = [true_human.w for true_human in true_humans]
            true_ws = torch.stack(true_ws)

            for t in range(self.args.sequence_length):

                sims = [SimulatedHuman(self.args, w=belief) for belief in beliefs[t]]

                belief_ws = [sim.w for sim in sims]
                belief_ws = torch.stack(belief_ws)

                mse = self.mse(belief_ws, true_ws)
                alignment = torch.tensor([true_humans[i].alignment(sims[i]) for i in range(self.args.batchsize)]).mean()

                mses.append(mse)
                alignments.append(alignment)

                if self.args.verbose:
                    print("Query %2d: MSE = %.3f, Alignment = %.3f" % (t, mse, alignment))

            # save plots for errors after pre training
            if self.args.visualize:
                plt.plot(mses)
                plt.xlabel("Queries")
                plt.ylabel("MSE")
                plt.title("Pretraining Evaluation - Reward Error")
                plt.savefig(self.dir + "eval-error")
                plt.close()

                plt.plot(alignments)
                plt.xlabel("Queries")
                plt.ylabel("Alignment")
                plt.title("Pretraining Evaluation - Reward Alignment")
                plt.savefig(self.dir + "eval-alignment")
                plt.close()
            
        print("########### TRAINING ###########")
        # alternate between training encoder and training policy
        # will also have to modify way we input to encoder to
        # take advantage of RNN structure
        # TODO

        for n in range(self.args.num_iters):
            
            # train encoder
            for _ in range(self.args.encoder_spi):
                # use policy in here
                pass

            # train policy
            self.policy.train_policy(n=self.args.policy_spi)
       

        
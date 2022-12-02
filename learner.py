from utils.helpers import collect_dataset, makedir
from models import encoder
from torch import optim, nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from query.simulate import response_dist, alignment

class Learner:

    def __init__(self, args, world, policy, exp_name="default"):
        self.args = args
        self.policy = policy
        self.exp_name = exp_name

        # get dataset
        self.dataset = collect_dataset(args, world)

         # initialize encoder network
        self.encoder = encoder.Encoder(args)

        # initialize optimizer for model
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=args.lr)

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

        #test if model can learn on only one true reward
        if self.args.batchsize == 1:
            true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

        for i in range(self.args.pretrain_len):
            if self.args.batchsize > 1:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)
            
            # get latents from queries and answers
            hidden = self.encoder.init_hidden()

            # manually handle hidden inputs for gru
            # compute loss and backprop at each step in sequence
            for t in range(self.args.sequence_length):
                self.optimizer.zero_grad()
                beliefs, hidden = self.encoder(query_seqs[t, :, :], answer_seqs[t, :], hidden)
                hidden = hidden.detach()

                # compute predicted response at each timestep for each sequence
                inputs = response_dist(self.args, query_seqs[t, :, :], beliefs)

                # get inputs and targets for cross entropy loss
                inputs = inputs.view(-1, self.args.query_size)
                targets = answer_seqs[t, :].view(-1)

                loss = self.loss(inputs, targets)
                loss.backward()

                if (t+1) % self.args.sequence_length == 0:
                    losses.append(loss.item())

                self.optimizer.step()

                if self.args.verbose and (i+1) % 100 == 0 and (t+1) % self.args.sequence_length == 0:
                    print("Iteration %2d: Loss = %.3f" % (i + 1, loss))

        # save plots for errors and losses after pre training
        if self.args.visualize:
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "train-loss")
            plt.close()

        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### PRE TRAINING - EVALUATION #########")

        if self.args.batchsize == 1:
            print(true_humans[0])
            
        with torch.no_grad():
            if self.args.batchsize > 1:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            test_losses = []
            mses = []
            alignments = []
            hidden = self.encoder.init_hidden()
            for t in range(self.args.sequence_length):
                beliefs, hidden = self.encoder(query_seqs[t, :, :], answer_seqs[t, :], hidden)
                print(beliefs[0])

                # compute predicted response at each timestep for each sequence
                inputs = response_dist(self.args, query_seqs[t, :, :], beliefs)

                # get inputs and targets for cross entropy loss
                inputs = inputs.view(-1, self.args.query_size)
                targets = answer_seqs[t, :].view(-1)

                loss = self.loss(inputs, targets)
                mse = self.mse(beliefs, true_humans)
                align = alignment(beliefs, true_humans).mean()

                test_losses.append(loss)
                mses.append(mse)
                alignments.append(align)

                if self.args.verbose:
                    print("Query %2d: Loss = %.3f, MSE = %.3f, Alignment = %.3f" % (t, loss, mse, align))

            # save plots for errors after pre training
            if self.args.visualize:
                plt.plot(test_losses)
                plt.xlabel("Queries")
                plt.ylabel("CE Loss")
                plt.title("Pretraining Evaluation - Loss")
                plt.savefig(self.dir + "eval-loss")
                plt.close()

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
       

        
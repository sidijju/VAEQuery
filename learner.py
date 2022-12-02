from utils.helpers import makedir
from models import encoder
from torch import optim, nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from query.simulate import response_dist, alignment

class Learner:

    def __init__(self, args, dataset, policy):
        self.args = args
        self.policy = policy
        self.exp_name = args.exp_name

        # get dataset
        self.dataset = dataset

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

    def pretrain(self):
        # pre-train encoder with untrained policy
        # only train encoder - do not use policy
        if self.args.verbose:
            print("######### PRE TRAINING #########")
            
        losses = []

        # test if model can learn on only one true reward
        # same reward used for testing and training
        if self.args.batchsize == 1:
            true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=1, seqlength=self.args.sequence_length)

        for i in range(self.args.pretrain_len):
            if self.args.batchsize > 1:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)
            
            self.optimizer.zero_grad()

            # get latents from queries and answers
            hidden = self.encoder.init_hidden()
            loss = 0

            # manually handle hidden inputs for gru for sequence
            for t in range(self.args.sequence_length):
                beliefs, hidden = self.encoder(query_seqs[t, :, :], answer_seqs[t, :], hidden)
                hidden = hidden.detach()

                # compute predicted response at timestep for each sequence
                inputs = response_dist(self.args, query_seqs[t, :, :], beliefs)

                # get inputs and targets for cross entropy loss
                inputs = inputs.view(-1, self.args.query_size)
                targets = answer_seqs[t, :].view(-1)

                seq_loss = self.loss(inputs, targets)
                loss += seq_loss
            
            loss /= self.args.sequence_length
            loss.backward()     
            losses.append(loss.item())           

            self.optimizer.step()                

            if self.args.verbose and (i+1) % 100 == 0:
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
            
        with torch.no_grad():
            # when we're not doing a singular batch experiment, we select a different test batch
            if self.args.batchsize > 1:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            test_losses = []
            mses = []
            alignments = []
            hidden = self.encoder.init_hidden()

            for t in range(self.args.sequence_length):
                beliefs, hidden = self.encoder(query_seqs[t, :, :], answer_seqs[t, :], hidden)

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

            if self.args.batchsize == 1:
                print("Reward Comparison")
                print("True  ", true_humans[0].data)
                print("Belief", beliefs[0].data)

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
            
    def train(self):
        # pretrain encoder if necessary
        if self.args.pretrain_len > 0:
            self.pretrain()

        if self.args.verbose:
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
       

        
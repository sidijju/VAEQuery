from utils.helpers import makedir
from models import encoder
from torch import optim, nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from query.simulate import *

class Learner:

    def __init__(self, args, dataset, policy):
        self.args = args
        self.dataset = dataset
        self.policy = policy(args, dataset)
        self.exp_name = args.exp_name

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

        # test if model can learn on only one true reward
        # same reward used for testing and training
        if self.args.one_reward:
            true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=1, seqlength=self.args.sequence_length)
            self.global_data = (true_humans, query_seqs, answer_seqs)
            self.args.batchsize = 1
        # TODO remove above later

    def pretrain(self): 
        # pre-train encoder with no policy input on whole sequences
        if self.args.verbose:
            print("######### PRE TRAINING #########")
            
        losses = []

        for i in range(self.args.pretrain_iters):
            if self.args.one_reward:
                # test if model can learn on only one true reward
                # same reward used for testing and training
                true_humans, query_seqs, answer_seqs = self.global_data
            else:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            # get latents from queries and answers
            hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)

            # manually handle hidden inputs for gru for sequence
            for t in range(self.args.sequence_length):
                self.optimizer.zero_grad()
                queries, answers = query_seqs[t, :, :].unsqueeze(0), answer_seqs[t, :, :].unsqueeze(0)

                beliefs, hidden = self.encoder(queries, answers, hidden)
                hidden = hidden.detach()

                # compute predicted response at timestep for each sequence
                inputs = response_dist(self.args, queries, beliefs)

                # get inputs and targets for cross entropy loss
                inputs = inputs.view(-1, self.args.query_size)
                targets = answer_seqs[t, :].view(-1)

                loss = self.loss(inputs, targets)
                loss.backward()     
                losses.append(loss.item())           

                self.optimizer.step()       

            if self.args.verbose and (i+1) % 100 == 0:
                print("Iteration %2d: Loss = %.3f" % (i + 1, loss))
                
        # save plots for losses after pre training
        if self.args.visualize:
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "pretrain-loss")
            plt.close()

        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### PRE TRAINING - EVALUATION #########")
            
        test_losses = []
        mses = []
        alignments = []
        
        with torch.no_grad():
            # when we're not doing a singular batch experiment, we select a different test batch
            if self.args.one_reward:
                true_humans, query_seqs, answer_seqs = self.global_data
            else:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)
            
            hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)

            for t in range(self.args.sequence_length):
                queries, answers = query_seqs[t, :, :].unsqueeze(0), answer_seqs[t, :, :].unsqueeze(0)

                beliefs, hidden = self.encoder(queries, answers, hidden)
                hidden = hidden.detach()

                # compute predicted response at timestep for each sequence
                inputs = response_dist(self.args, queries, beliefs)

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

            if self.args.one_reward:
                print("Reward Comparison")
                print("True  ", true_humans[0].data)
                print("Belief", beliefs[0].data)

        # save plots for errors after pre training
        if self.args.visualize:
            plt.plot(test_losses)
            plt.xlabel("Queries")
            plt.ylabel("CE Loss")
            plt.title("Pretraining Evaluation - Loss")
            plt.savefig(self.dir + "preeval-loss")
            plt.close()

            plt.plot(mses)
            plt.xlabel("Queries")
            plt.ylabel("MSE")
            plt.title("Pretraining Evaluation - Reward Error")
            plt.savefig(self.dir + "preeval-error")
            plt.close()

            plt.plot(alignments)
            plt.xlabel("Queries")
            plt.ylabel("Alignment")
            plt.title("Pretraining Evaluation - Reward Alignment")
            plt.savefig(self.dir + "preeval-alignment")
            plt.close()
            
    def train(self):
        # pretrain encoder if necessary
        if self.args.pretrain_iters > 0:
            self.pretrain()

        if self.args.verbose:
            print("########### TRAINING ###########")

        losses = []

        for n in range(self.args.num_iters):
            # get batch of starting queries for iteration
            # same reward used for testing and training
            if self.args.one_reward:
                true_humans, query_seqs, answer_seqs = self.global_data
                queries, answers = query_seqs[0], answer_seqs[0]
            else:
                true_humans, queries, answers = self.dataset.get_batch(batchsize=self.args.batchsize)

            # train encoder
            for _ in range(self.args.encoder_spi):
                
                # initialize hidden and loss variables
                self.optimizer.zero_grad()
                hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)
                loss = 0

                # manually handle hidden inputs for gru for sequence
                curr_queries = queries.unsqueeze(0)
                curr_answers = answers.unsqueeze(0)

                for _ in range(self.args.sequence_length):

                    # get beliefs from queries and answers
                    beliefs, hidden = self.encoder(curr_queries, curr_answers, hidden)
                    hidden = hidden.detach()

                    # compute predicted response at timestep for each query
                    inputs = response_dist(self.args, curr_queries, beliefs)

                    # get inputs and targets for cross entropy loss
                    inputs = inputs.view(-1, self.args.query_size)
                    targets = curr_answers.view(-1)

                    # compute loss and back propagate
                    loss = self.loss(inputs, targets)
                    loss.backward()     
                    losses.append(loss.item())    
                    self.optimizer.step()   

                    # # get next queries
                    # curr_queries = self.policy.run_policy(curr_queries, beliefs)
                    # # get next answers (in the case of an actual human, would be replaced with labeling step)
                    # answer_dist = response_dist(self.args, curr_queries, true_humans)
                    # curr_answers = sample_dist(self.args, answer_dist)

                    # # reshape to have sequence dim of 1
                    # curr_queries = curr_queries.unsqueeze(0)
                    # curr_answers = curr_answers.unsqueeze(0)             

            if self.args.verbose and (n+1) % 100 == 0:
                print("Iteration %2d: Loss = %.3f" % (n+1, losses[-1]))

            # train policy
            self.policy.train_policy(n=self.args.policy_spi)
       
        # save plots for losses after training
        if self.args.visualize:
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "train-loss")
            plt.close()
        
        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### TRAINING - EVALUATION #########")
            
        test_losses = []
        mses = []
        alignments = []
        
        with torch.no_grad():
            # when we're not doing a singular batch experiment, we select a different test batch
            # same reward used for testing and training
            if self.args.one_reward:
                true_humans, query_seqs, answer_seqs = self.global_data
                queries, answers = query_seqs[0], answer_seqs[0]
            else:
                true_humans, queries, answers = self.dataset.get_batch(batchsize=self.args.batchsize)

            # initialize hidden and loss variables
            hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)

            # manually handle hidden inputs for gru for sequence
            curr_queries = queries.unsqueeze(0)
            curr_answers = answers.unsqueeze(0)

            for _ in range(self.args.sequence_length):

                # get beliefs from queries and answers
                beliefs, hidden = self.encoder(curr_queries, curr_answers, hidden)
                hidden = hidden.detach()

                # compute predicted response at timestep for each query
                inputs = response_dist(self.args, curr_queries, beliefs)

                # get inputs and targets for cross entropy loss
                inputs = inputs.view(-1, self.args.query_size)
                targets = curr_answers.view(-1)

                # compute metrics and store in lists
                loss = self.loss(inputs, targets)
                mse = self.mse(beliefs, true_humans)
                align = alignment(beliefs, true_humans).mean()

                test_losses.append(loss.item())
                mses.append(mse)
                alignments.append(align)

                # # get next queries
                # curr_queries = self.policy.run_policy(curr_queries, beliefs)
                # # get next answers (in the case of an actual human, would be replaced with labeling step)
                # answer_dist = response_dist(self.args, curr_queries, true_humans)
                # curr_answers = sample_dist(self.args, answer_dist)

                # # reshape to have sequence dim of 1
                # curr_queries = curr_queries.unsqueeze(0)
                # curr_answers = curr_answers.unsqueeze(0)   

                if self.args.verbose:
                    print("Query %2d: Loss = %.3f, MSE = %.3f, Alignment = %.3f" % (t, loss, mse, align))

            if self.args.one_reward:
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
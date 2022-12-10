from utils.helpers import makedir
from models import encoder
from torch import optim, nn
import torch
import matplotlib.pyplot as plt
from query.simulate import *

class Learner:

    def __init__(self, args, datasets, policy):
        self.args = args
        self.train_dataset, self.val_dataset, self.test_dataset = datasets
        self.policy = policy(args)
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
            true_humans, query_seqs, answer_seqs = self.train_dataset.get_batch_seq(batchsize=1, seqlength=self.args.sequence_length)
            self.global_data = (true_humans, query_seqs, answer_seqs)
            self.args.batchsize = 1
        # TODO remove above later

    def pretrain(self): 
        # pre-train encoder with no policy input on whole sequences
        if self.args.verbose:
            print("######### PRE TRAINING #########")
            
        losses = []
        # set model to train mode
        self.encoder.train()

        for i in range(self.args.pretrain_iters):
            if self.args.one_reward:
                # test if model can learn on only one true reward
                # same reward used for testing and training
                true_humans, query_seqs, answer_seqs = self.global_data
            else:
                true_humans, query_seqs, answer_seqs = self.train_dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            # initialize hidden and loss variables
            hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)
            loss = 0

            # manually handle hidden inputs for gru for sequence
            for t in range(self.args.sequence_length):
                queries, answers = query_seqs[t, :, :].unsqueeze(0), answer_seqs[t, :, :].unsqueeze(0)

                beliefs, hidden = self.encoder(queries, answers, hidden)

                # compute predicted response at timestep for each sequence
                inputs = response_dist(self.args, queries, beliefs)

                # get inputs and targets for cross entropy loss
                inputs = inputs.view(-1, self.args.query_size)
                targets = answer_seqs[t, :].view(-1)

                loss += self.loss(inputs, targets)
            
            self.optimizer.zero_grad()
            loss /= self.args.sequence_length
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
            # set model to eval mode
            self.encoder.eval()

            # when we're not doing a singular batch experiment, we select a different test batch
            if self.args.one_reward:
                true_humans, query_seqs, answer_seqs = self.global_data
            else:
                true_humans, query_seqs, answer_seqs = self.train_dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)
            
            hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)

            for t in range(self.args.sequence_length):
                queries, answers = query_seqs[t, :, :].unsqueeze(0), answer_seqs[t, :, :].unsqueeze(0)

                beliefs, hidden = self.encoder(queries, answers, hidden)

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
        val_losses = []
        # set model to train mode
        self.encoder.train()

        for n in range(self.args.num_iters):
            # get batch of starting queries for iteration
            # same reward used for testing and training
            if self.args.one_reward:
                true_humans, query_seqs, answer_seqs = self.global_data
                queries, answers = query_seqs[0], answer_seqs[0]
                val_queries, val_answers = query_seqs[-1], answer_seqs[-1]
                val_humans = true_humans
            else:
                true_humans, queries, answers = self.train_dataset.get_batch(batchsize=self.args.batchsize)
                val_humans, val_queries, val_answers = self.val_dataset.get_batch(batchsize=self.args.batchsize)

            # train encoder
            for _ in range(self.args.encoder_spi):

                # initialize storage for entire iteration
                query_seqs = torch.zeros((self.args.sequence_length, *(queries.shape)))
                answer_seqs = torch.zeros((self.args.sequence_length, *(answers.shape))).to(torch.long)
                val_query_seqs = torch.zeros_like(query_seqs)
                val_answer_seqs = torch.zeros_like(answer_seqs).to(torch.long)

                # set first query in sequences to the batch
                query_seqs[0] = queries
                answer_seqs[0] = answers
                val_query_seqs[0] = val_queries
                val_answer_seqs[0] = val_answers

                for t in range(self.args.sequence_length):

                    # initialize loss for the iteration
                    loss = 0
                    val_loss = 0

                    for i in range(t+1):

                        # manually handle hidden inputs for gru for sequence
                        curr_queries = query_seqs[i].unsqueeze(0)
                        curr_answers = answer_seqs[i].unsqueeze(0)
                        val_queries = val_query_seqs[i].unsqueeze(0)
                        val_answers = val_answer_seqs[i].unsqueeze(0)

                        # initialize hidden variables
                        hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)
                        val_hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)
                    
                        # get beliefs from queries and answers
                        beliefs, hidden = self.encoder(curr_queries, curr_answers, hidden)
                        val_beliefs, val_hidden = self.encoder(val_queries, val_answers, val_hidden)

                        # compute predicted response at timestep for each query
                        inputs = response_dist(self.args, curr_queries, beliefs)
                        val_inputs = response_dist(self.args, val_queries, val_beliefs)

                        # get inputs and targets for cross entropy loss
                        inputs = inputs.view(-1, self.args.query_size)
                        val_inputs = val_inputs.view(-1, self.args.query_size)
                        targets = curr_answers.view(-1)
                        val_targets = val_answers.view(-1)

                        # compute loss so far
                        loss += self.loss(inputs, targets)
                        val_loss += self.loss(val_inputs, val_targets)

                    # optimize over the current sequence
                    self.optimizer.zero_grad()

                    ## NOT SURE IF I SHOULD DO THIS ##
                    loss /= (t+1)
                    val_loss /= (t+1)
                    ##################################
                    
                    loss.backward()       
                    self.optimizer.step()  

                    if t+1 < self.args.sequence_length:
                        # get next queries
                        next_queries = self.policy.run_policy(query_seqs[t], beliefs, self.train_dataset)
                        next_val_queries = self.policy.run_policy(val_query_seqs[t], val_beliefs, self.val_dataset)

                        # get next answers (in the case of an actual human, would be replaced with labeling step)
                        next_answers = sample_dist(self.args, response_dist(self.args, curr_queries, true_humans))
                        next_val_answers = sample_dist(self.args, response_dist(self.args, val_queries, val_humans))

                        # add next queries to sequence
                        query_seqs[t+1] = next_queries
                        answer_seqs[t+1] = next_answers    
                        val_query_seqs[t+1] = next_val_queries
                        val_answer_seqs[t+1] = next_val_answers

                # store losses
                losses.append(loss.item())  
                val_losses.append(val_loss.item())  

            if self.args.verbose and (n+1) % 100 == 0:
                print("Iteration %2d: Loss = %.3f, Val Loss = %.3f" % (n+1, losses[-1], val_losses[-1]))

            # train policy
            self.policy.train_policy(self.train_dataset, n=self.args.policy_spi)
       
        # save plots for losses after training
        if self.args.visualize:
            plt.plot(losses, label='train')
            plt.plot(val_losses, label='validation')
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "train-loss")
            plt.close()

    def test(self):
        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### TESTING #########")

        # set the encoder to eval mode
        self.encoder.eval()
            
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
                true_humans, queries, answers = self.test_dataset.get_batch(batchsize=self.args.batchsize)

            # initialize hidden and loss variables
            hidden = self.encoder.init_hidden(batchsize=self.args.batchsize)

            # manually handle hidden inputs for gru for sequence
            curr_queries = queries.unsqueeze(0)
            curr_answers = answers.unsqueeze(0)

            for t in range(self.args.sequence_length):

                print()
                print()

                # get beliefs from queries and answers
                beliefs, hidden = self.encoder(curr_queries, curr_answers, hidden)

                print("Beliefs", beliefs)
                print(torch.norm(beliefs, dim=-1))
                print("True Humans", true_humans)

                # get inputs and targets for cross entropy loss
                inputs = response_dist(self.args, curr_queries, beliefs)
                inputs = inputs.view(-1, self.args.query_size)
                targets = curr_answers.view(-1)

                from torch.nn.functional import softmax
                print("Dist", softmax(inputs, dim=-1))
                print("Targets", targets)

                # compute metrics and store in lists
                loss = self.loss(inputs, targets)
                mse = self.mse(beliefs, true_humans)
                print(alignment(beliefs, true_humans))
                align = alignment(beliefs, true_humans).mean()

                test_losses.append(loss.item())
                mses.append(mse)
                alignments.append(align)

                # get next queries
                curr_queries = self.policy.run_policy(curr_queries, beliefs, self.test_dataset)
                # get next answers (in the case of an actual human, would be replaced with labeling step)
                answer_dist = response_dist(self.args, curr_queries, true_humans)
                curr_answers = sample_dist(self.args, answer_dist)

                # reshape to have sequence dim of 1
                curr_queries = curr_queries.unsqueeze(0)
                curr_answers = curr_answers.unsqueeze(0)
                    
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
            plt.savefig(self.dir + "test-loss")
            plt.close()

            plt.plot(mses)
            plt.xlabel("Queries")
            plt.ylabel("MSE")
            plt.title("Pretraining Evaluation - Reward Error")
            plt.savefig(self.dir + "test-error")
            plt.close()

            plt.plot(alignments)
            plt.xlabel("Queries")
            plt.ylabel("Alignment")
            plt.title("Pretraining Evaluation - Reward Alignment")
            plt.savefig(self.dir + "test-alignment")
            plt.close()
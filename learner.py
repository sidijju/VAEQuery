from utils.helpers import makedir, reparameterize
from models import encoder
from torch import optim, nn
import torch
import matplotlib.pyplot as plt
from query.simulate import *
from storage.vae_storage import *
from tqdm import trange

class Learner:

    def __init__(self, args, datasets, policy):
        self.args = args
        self.batchsize = self.args.batchsize
        self.train_dataset, self.test_dataset = datasets
        self.policy = policy(args)
        self.exp_name = args.exp_name

         # initialize encoder network
        self.encoder = encoder.Encoder(args)

        # initialize optimizer for model
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=args.lr)

        # define loss functions for VAE
        self.loss = nn.CrossEntropyLoss(reduction = 'sum')
        self.mse = nn.MSELoss(reduction='none')

        # directory for logs
        self.dir = self.args.log_dir + self.policy.vis_directory
        makedir(self.dir)

    def pretrain(self): 
        # pre-train encoder with no policy input on whole sequences
        if self.args.verbose:
            print("######### PRE TRAINING #########")
            
        losses = []
        # set model to train mode
        self.encoder.train()

        for i in range(self.args.pretrain_iters):
            true_humans, query_seqs, answer_seqs = self.train_dataset.get_batch_seq(batchsize=self.batchsize, seqlength=self.args.sequence_length)

            # initialize hidden and loss variables
            hidden = self.encoder.init_hidden(batchsize=self.batchsize)
            loss = 0

            # manually handle hidden inputs for gru for sequence
            for t in range(self.args.sequence_length):
                queries, answers = query_seqs[t].unsqueeze(0), answer_seqs[t].unsqueeze(0)

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

            true_humans, query_seqs, answer_seqs = self.train_dataset.get_batch_seq(batchsize=self.batchsize, seqlength=self.args.sequence_length)
            hidden = self.encoder.init_hidden(batchsize=self.batchsize)

            for t in range(self.args.sequence_length):
                queries, answers = query_seqs[t].unsqueeze(0), answer_seqs[t].unsqueeze(0)

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
        # set model to train mode
        self.encoder.train()

        for n in trange(self.args.num_iters):

            # train encoder
            for _ in range(self.args.encoder_spi):

                # get batch of starting queries for iteration
                true_humans, queries, answers = self.train_dataset.get_batch(batchsize=self.batchsize)
                queries = order_queries(queries, answers)

                # initialize storage for entire iteration
                query_seqs = torch.zeros((self.args.sequence_length, *(queries.shape)))
                mus = torch.zeros((self.args.sequence_length, self.batchsize, self.args.num_features))
                logvars = torch.zeros_like(mus)

                # set first query in sequences to the batch
                query_seqs[0] = queries

                # initialize hidden variables
                hidden = self.encoder.init_hidden(batchsize=self.batchsize)

                # compute query sequence using encoder
                for t in range(self.args.sequence_length):
                    # get beliefs from queries and answers
                    mus[t], logvars[t], hidden = self.encoder(query_seqs[t].clone().unsqueeze(0), hidden)
                
                    if t+1 < self.args.sequence_length:
                        # get next queries
                        next_queries = self.policy.run_policy(mus[t], logvars[t], self.train_dataset)
                        next_answers = respond_queries(self.args, next_queries, true_humans)
                        next_queries = order_queries(next_queries, next_answers)

                         # add next queries to sequence
                        query_seqs[t+1] = next_queries 

                # get batch of random queries for loss computations 
                _, loss_queries, loss_answers = self.train_dataset.get_batch(batchsize=self.batchsize, true_rewards=true_humans)

                # initialize loss variables
                loss = 0
                
                # compute losses over sequence
                for t in range(self.args.sequence_length):
                    # compute predicted response for each of the loss queries
                    sample = reparameterize(self.args, mus[t], logvars[t])
                    sample = sample.squeeze(1)
                    inputs = response_dist(self.args, loss_queries, sample)
                
                    # get inputs for cross entropy loss
                    inputs = inputs.view(-1, self.args.query_size)

                    # get targets for cross entropy loss
                    targets = loss_answers.view(-1)
                    
                    # compute and aggregate loss
                    loss += (t+1) * self.loss(inputs, targets)
                    loss += (t+1) * torch.sum(torch.square(torch.linalg.norm(mus[t], dim=-1) - 1))
                    
                # optimize over iteration
                self.optimizer.zero_grad()
                loss /= self.batchsize
                loss.backward()    
                self.optimizer.step()  

                # store losses
                losses.append(loss.item())  

            if self.args.verbose and (n+1) % 10 == 0:
                print("\nIteration %2d: Loss = %.3f" % (n+1, losses[-1]))

            # train policy
            self.policy.train_policy(self.train_dataset, n=self.args.policy_spi)
       
        # save plots for losses after training
        if self.args.visualize:
            plt.plot(losses, label='train')
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "train-loss")
            plt.close()

        return losses

    def test(self):
        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### TESTING #########")

        # set the encoder to eval mode
        self.encoder.eval()
            
        test_losses = []
        mses_mean = []
        mses_std = []
        alignments_mean = []
        alignments_std = []
        
        with torch.no_grad():
            true_humans, queries, answers = self.test_dataset.get_batch(batchsize=self.batchsize)
            queries = order_queries(queries, answers)
            hidden = self.encoder.init_hidden(batchsize=self.batchsize)

            for t in range(self.args.sequence_length):

                # get beliefs from queries and answers
                mu, logvar, hidden = self.encoder(queries.clone().unsqueeze(0), hidden)

                # get inputs and targets for cross entropy loss
                sample = reparameterize(self.args, mu, logvar)
                sample = sample.squeeze(1)
                inputs = response_dist(self.args, queries, sample)
                inputs = inputs.view(-1, self.args.query_size)
                targets = answers.view(-1)

                # compute metrics and store in lists
                loss = self.loss(inputs, targets) / self.batchsize
                mses = torch.sum(self.mse(sample, true_humans), dim=-1) / self.batchsize
                align = alignment(sample, true_humans)

                test_losses.append(loss.item())
                mses_mean.append(mses.mean())
                mses_std.append(mses.std())
                alignments_mean.append(align.mean())
                alignments_std.append(align.std())

                # get next queries
                queries = self.policy.run_policy(mu, logvar, self.test_dataset)
                # get next answers (in the case of an actual human, would be replaced with labeling step)
                answers = respond_queries(self.args, queries, true_humans)
                # reorder queries
                queries = order_queries(queries, answers)
                    
                if self.args.verbose:
                    print("Query %2d: Loss = %5.3f, MSE = %5.3f, Alignment = %5.3f" % (t, loss, mses_mean[-1], alignments_mean[-1]))

        # save plots for errors after pre training
        if self.args.visualize:
            plt.plot(test_losses)
            plt.xlabel("Queries")
            plt.ylabel("CE Loss")
            plt.title("Test Evaluation - Loss")
            plt.savefig(self.dir + "test-loss")
            plt.close()
            plt.errorbar(range(1, len(mses_mean)+1), mses_mean, yerr=mses_std/(np.sqrt(len(mses_std))))
            plt.xlabel("Queries")
            plt.xticks(range(1, self.args.sequence_length+1))
            plt.ylabel("MSE")
            plt.title("Test Evaluation - Reward Error")
            plt.savefig(self.dir + "test-error")
            plt.close()
            plt.errorbar(range(1, len(alignments_mean)+1), alignments_mean, yerr=alignments_std/(np.sqrt(len(alignments_std))))
            plt.xlabel("Queries")
            plt.xticks(range(1, self.args.sequence_length+1))
            plt.ylabel("Alignment")
            plt.title("Test Evaluation - Reward Alignment")
            plt.savefig(self.dir + "test-alignment")
            plt.close()

        # save encoder model
        torch.save(self.encoder.state_dict(), self.dir + "model.pt")

        # return test results
        return mses_mean, alignments_mean
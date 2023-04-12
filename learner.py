from utils.helpers import makedir, reparameterize
from models import encoder
from torch import optim, nn
import torch
import matplotlib.pyplot as plt
from query.simulate import *
from storage.vae_storage import *
from tqdm import trange
from policies.policies import GreedyApproximationPolicy

class Learner:

    def __init__(self, args, datasets, policy, encoder_start=None):
        self.args = args
        self.batchsize = self.args.batchsize
        self.train_dataset, self.test_dataset = datasets
        self.exp_name = args.exp_name

        # initialize encoder network if not given
        if not encoder_start:
            self.encoder = encoder.Encoder(args)
            self.encoder.to(self.args.device)
        else:
            self.encoder = encoder_start
            self.encoder.to(self.args.device)

        # initialize policy
        self.policy = policy(args, self.encoder, self.train_dataset)

        # initialize optimizer for model
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=args.lr)

        # define loss functions for VAE
        self.loss = nn.CrossEntropyLoss(reduction = 'sum')
        self.mse = nn.MSELoss()

        # directory for logs
        self.dir = self.args.log_dir + self.policy.vis_directory
        makedir(self.dir)

    ### get query dataset for meta-iteration ###
    def compute_sequence(self):
        # get batch of starting queries for iteration
        true_humans, queries, answers = self.train_dataset.get_batch(batchsize=self.batchsize)
        queries = order_queries(queries, answers)

        # initialize storage for entire iteration
        query_seqs = torch.zeros((self.args.sequence_length, *(queries.shape))).to(self.args.device)
        mus = torch.zeros((self.args.sequence_length, self.batchsize, self.args.num_features)).to(self.args.device)
        logvars = torch.zeros_like(mus).to(self.args.device)

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

        return query_seqs, mus, logvars, true_humans
    
    def compute_loss_over_sequence(self, query_seqs, mus, logvars, true_humans):
        it_mus = torch.zeros_like(mus).to(self.args.device)
        it_logvars = torch.zeros_like(logvars).to(self.args.device)

        # initialize hidden variables
        hidden = self.encoder.init_hidden(batchsize=self.batchsize)

        # compute query sequence using encoder
        for t in range(self.args.sequence_length):
            # get beliefs from queries and answers
            it_mus[t], it_logvars[t], hidden = self.encoder(query_seqs[t].clone().unsqueeze(0), hidden)

        # get batch of random queries for loss computations 
        _, loss_queries, loss_answers = self.train_dataset.get_batch(batchsize=self.batchsize, true_rewards=true_humans)

        # initialize loss variables
        loss = 0
        
        # compute losses over sequence
        for t in range(self.args.sequence_length):
            # compute predicted response for each of the loss queries
            sample = reparameterize(self.args, it_mus[t], it_logvars[t])
            sample = sample.squeeze(1)
            inputs = response_dist(self.args, loss_queries, sample)
        
            # get inputs for cross entropy loss
            inputs = inputs.view(-1, self.args.query_size)

            # get targets for cross entropy loss
            targets = loss_answers.view(-1)
            
            # compute and aggregate loss
            loss += (t+1) * self.loss(inputs, targets)
            loss += (t+1) * torch.sum(torch.square(torch.linalg.norm(it_mus[t], dim=-1) - 1))

        return loss

    def train(self):

        if self.args.verbose:
            print("########### TRAINING ###########")

        losses = []
        # set model to train mode
        self.encoder.train()

        if self.args.increasing_policy_spi:
            bounds = self.args.increasing_policy_spi
            policy_spi_schedule = np.linspace(bounds[0], bounds[1], num=self.args.num_iters, dtype=int)
        else:
            policy_spi_schedule = [self.args.policy_spi] * self.args.num_iters

        if isinstance(self.policy, GreedyApproximationPolicy):

            self.policy.train_policy(0, n=policy_spi_schedule[0])

            for n in trange(self.args.num_iters):

                query_seqs, mus, logvars, true_humans = self.compute_sequence()

                for _ in range(self.args.encoder_spi):

                    loss = self.compute_loss_over_sequence(query_seqs, mus, logvars, true_humans)
                    
                    # train encoder if not given previously
                    if self.train_encoder:
                        self.encoder.train()
                        self.optimizer.zero_grad()
                        loss.backward()    
                        self.optimizer.step()

                    # store losses
                    losses.append(loss.item())  

                if self.args.verbose and (n+1) % 10 == 0:
                    print("\nIteration %2d: Loss = %.3f" % (n+1, losses[-1]))

        else:
            for n in trange(self.args.num_iters):

                # train policy
                self.policy.train_policy(n, n=policy_spi_schedule[n])

                query_seqs, mus, logvars, true_humans = self.compute_sequence()

                # train encoder if not given previously
                if not self.args.random_encoder:
                    for _ in range(self.args.encoder_spi):

                        loss = self.compute_loss_over_sequence(query_seqs, mus, logvars, true_humans)
                            
                        # optimize over iteration
                        self.optimizer.zero_grad()
                        loss /= self.batchsize
                        loss.backward()    
                        self.optimizer.step()  

                        # store losses
                        losses.append(loss.item())  

                if self.args.verbose and (n+1) % 10 == 0:
                    print("\nIteration %2d: Loss = %.3f" % (n+1, losses[-1]))
        
        # save plots for losses after training
        if self.args.visualize:
            plt.plot(losses, label='train')
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "train-loss")
            plt.close()

        # save models
        torch.save(self.encoder.state_dict(), self.dir + "model.pt")
        # TODO fix later
        if hasattr(self.policy, "model"):
            self.policy.model.save(self.policy.log_dir + "model")

        return losses

    def test(self, batch=None):
        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### TESTING #########")

        # set the encoder to eval mode
        self.encoder.eval()
            
        test_losses = []
        mses_mean = []
        alignments_mean = []
        
        with torch.no_grad():

            if batch:
                true_humans, orig_queries, answers = batch
                queries = order_queries(orig_queries, answers)
            else:
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
                mses = self.mse(sample, true_humans)

                # TODO do this over a batch of samples
                align = alignment(sample, true_humans)

                test_losses.append(loss.item())
                mses_mean.append(mses.mean().cpu())
                alignments_mean.append(align.mean().cpu())

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
            plt.plot(range(1, len(mses_mean)+1), mses_mean)
            plt.xlabel("Queries")
            plt.xticks(range(1, self.args.sequence_length+1))
            plt.ylabel("MSE")
            plt.title("Test Evaluation - Reward Error")
            plt.savefig(self.dir + "test-error")
            plt.close()
            plt.plot(range(1, len(alignments_mean)+1), alignments_mean)
            plt.xlabel("Queries")
            plt.xticks(range(1, self.args.sequence_length+1))
            plt.ylabel("Alignment")
            plt.title("Test Evaluation - Reward Alignment")
            plt.savefig(self.dir + "test-alignment")
            plt.close()

        # return test results
        return mses_mean, alignments_mean
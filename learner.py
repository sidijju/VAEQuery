from utils.helpers import collect_dataset, makedir
from models import encoder, belief
from torch import optim, nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from query.simulate import response_dist, alignment

class Learner:

    def plot_grad_flow(self, named_parameters, title=""):
        from matplotlib.lines import Line2D
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(title)
        plt.close()

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
        self.params = list(self.encoder.parameters()) + list(self.belief.parameters())
        self.named_params = list(self.encoder.named_parameters()) + list(self.belief.named_parameters())

        self.optimizer = optim.Adam(self.params, lr=args.lr)

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

            self.optimizer.zero_grad()
            self.encoder.init_hidden()

            # get latents from queries and answers
            latents = self.encoder(query_seqs, answer_seqs)
            assert latents.shape == (self.args.sequence_length, self.args.batchsize, self.args.latent_dim)

            # pass latent from all timesteps into belief network
            beliefs, _, _ = self.belief(latents)
            assert beliefs.shape == (self.args.sequence_length, self.args.batchsize, self.args.num_features)

            # compute predicted response at each timestep for each sequence
            inputs = response_dist(self.args, query_seqs, beliefs)

            # get inputs and targets for cross entropy loss
            inputs = inputs.view(-1, self.args.query_size)
            targets = answer_seqs.view(-1)

            loss = self.loss(inputs, targets)
            assert loss.shape == ()

            loss.backward()
            self.optimizer.step()

            if self.args.verbose and (i+1) % 100 == 0:
                print("Iteration %2d: Loss = %.3f" % (i+1, loss))
                losses.append(loss.item())

        # save plots for errors and losses after pre training
        if self.args.visualize:
            plt.plot(losses)
            plt.xlabel("Iterations (100s)")
            plt.ylabel("CE Error")
            plt.title("Query Distribution vs. Answer Error")
            plt.savefig(self.dir + "loss")
            plt.close()

        # evaluate encoder on test batch
        if self.args.verbose:
            print("######### PRE TRAINING - EVALUATION #########")

        if self.args.batchsize == 1:
            print(true_humans[0])
            
        with torch.no_grad():
            if self.args.batchsize > 1:
                true_humans, query_seqs, answer_seqs = self.dataset.get_batch_seq(batchsize=self.args.batchsize, seqlength=self.args.sequence_length)

            self.encoder.init_hidden()
            latents = self.encoder(query_seqs, answer_seqs)

            # latents should have the output at every timestep for the input sequence
            assert latents.shape == (self.args.sequence_length, self.args.batchsize, self.args.latent_dim)

            # pass latents into belief network
            beliefs, _, _ = self.belief(latents)

            assert beliefs.shape == (self.args.sequence_length, self.args.batchsize, self.args.num_features)

            mses = []
            alignments = []

            for t in range(self.args.sequence_length):

                print(beliefs[t, :, :].data)

                mse = self.mse(beliefs[t, :, :], true_humans)
                align = alignment(beliefs[t, :, :], true_humans).mean()

                mses.append(mse)
                alignments.append(align)

                if self.args.verbose:
                    print("Query %2d: MSE = %.3f, Alignment = %.3f" % (t, mse, align))

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
       

        
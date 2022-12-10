import argparse
from pathlib import Path

def args(rest_args):
    parser = argparse.ArgumentParser()

    ##### GENERAL #####

    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_dir', type=Path, default=Path('/logs'))
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--one_reward', dest='one_reward', action='store_true')
    parser.set_defaults(one_reward=False)
    parser.add_argument('--optimal_user', dest='optimal_user', action='store_true',
                        help='set user to be optimal based on probability distributions')
    parser.set_defaults(optimal_user=False)

    parser.add_argument('--temperature', type=float, default=1,
                        help='Boltzmann rationality temperature')
    parser.add_argument('--m', type=int, default=100,
                        help='how many samples for sampling from the belief distribution')
   
    ##### POLICY #####

    parser.add_argument('--query_size', type=int, default=2,
                        help='number of trajectories to choose from in one query')
    parser.add_argument('--policy_spi', type=int, default=1,
                        help='number of training steps per iteration for the policy')

    ##### VAE #####
    parser.add_argument('--encoder_spi', type=int, default=1,
                        help='number of training steps per iteration for the encoder')
    parser.add_argument('--precollect_num', type=int, default=1000,
                        help='how many trajectories to pre-collect before training begins')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='how many trajectories to keep in VAE buffer')
    parser.add_argument('--pretrain_iters', type=int, default=1000,
                        help='how many iterations to pretrain the vae')
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--batchsize', type=int, default=10,
                        help='how many queries to use for VAE update')
    parser.add_argument('--sequence_length', type=int, default=20,
                        help='how long a query sequence is for RNN')
    parser.add_argument('--fc_dim', type=int, default=16, 
                        help='dimensionality of fc input to GRU')
    parser.add_argument('--gru_hidden_layers', type=int, default=1, 
                        help='number of hidden layers')
    parser.add_argument('--gru_hidden_size', type=int, default=64, 
                        help='size of hidden layers')
    parser.add_argument('--fc_layers', type=list, default=[16, 32, 64], 
                        help='fc layers for VAE')
    parser.add_argument('--use_mu', dest='sample_belief', action='store_false')
    parser.set_defaults(sample_belief=True)

    ##### ENVIRONMENT #####

    parser.add_argument('--grid_size', type=int, default=5,
                        help='size of grid')
    parser.add_argument('--max_trajectory_len', type=int, default=15,
                        help='maximum length of trajectory')

    return parser.parse_args(rest_args)
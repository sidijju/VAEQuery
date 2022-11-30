import argparse
from pathlib import Path

def args(rest_args):
    parser = argparse.ArgumentParser()

    ##### GENERAL #####

    parser.add_argument('--num_iters', type=int, default=2e5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_dir', type=Path, default=Path('/logs'))
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--temperature', type=int, default=10,
                        help='Boltzmann rationality temperature')

    ##### POLICY #####

    #TODO
    parser.add_argument('--query_size', type=int, default=2,
                        help='number of trajectories to choose from in one query')

    ##### VAE #####
    parser.add_argument('--precollect_num', type=int, default=100,
                        help='how many trajectories to pre-collect before training begins')
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='how many trajectories to keep in VAE buffer')
    parser.add_argument('--pretrain_len', type=int, default=1000,
                        help='how many iterations to pretrain the vae')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batchsize', type=int, default=25,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--sequence_length', type=int, default=20,
                        help='how long a query sequence is for RNN')
    parser.add_argument('--max_trajectory_len', type=int, default=15,
                        help='maximum length of trajectory')
    parser.add_argument('--fc_dim', type=int, default=1, 
                        help='dimensionality of fc input to GRU')
    parser.add_argument('--gru_hidden_layers', type=int, default=1, 
                        help='number of hidden layers')
    parser.add_argument('--gru_hidden_size', type=int, default=64, 
                        help='size of hidden layers')
    parser.add_argument('--latent_dim', type=int, default=10, 
                        help='dimensionality of latent space')
    parser.add_argument('--fc_layers', type=list, default=[16, 32, 64], 
                        help='fc layers for VAE')
    

    ##### ENVIRONMENT #####

    parser.add_argument('--grid_size', type=int, default=5,
                        help='size of grid')

    return parser.parse_args(rest_args)
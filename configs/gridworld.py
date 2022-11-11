import argparse
from pathlib import Path

def args(rest_args):
    parser = argparse.ArgumentParser()

    ##### GENERAL #####

    parser.add_argument('--num_iters', type=int, default=2e5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_dir', type=Path, default=Path('/logs'))
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--precollect_len', type=int, default=5000,
                        help='how many steps to pre-collect before training begins')

    ##### POLICY #####

    #TODO

    ##### VAE #####

    parser.add_argument('--vae_lr', type=float, default=0.001)
    parser.add_argument('--vae_buffer_size', type=int, default=100000,
                        help='how many trajectories to keep in VAE buffer')
    parser.add_argument('--vae_pretrain_len', type=int, default=5000,
                        help='how many steps to pretrain the vae')
    parser.add_argument('--vae_buffer_add_thresh', type=float, default=1,
                        help='probability of adding a new trajectory to buffer')
    parser.add_argument('--vae_batchsize', type=int, default=25,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--vae_max_trajectory_len', type=int, default=60,
                        help='maximum length of trajectory')
    parser.add_argument('--vae_encoder_layers_before_gru', nargs='+', type=int, default=[])
    parser.add_argument('--vae_encoder_gru_hidden_size', type=int, default=64, 
                        help='dimensionality of RNN hidden state')
    parser.add_argument('--vae_encoder_layers_after_gru', nargs='+', type=int, default=[])
    parser.add_argument('--vae_latent_dim', type=int, default=5, 
                        help='dimensionality of latent space')

    return parser.parse_args(rest_args)
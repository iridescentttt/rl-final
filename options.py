#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--episodes", type=int, default=1500000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1027)
    parser.add_argument('--env_name', default="Pendulum-v0",
                        help='name of the environment to run')
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--hfeat1', default=400, type=int,
                        help='hidden num of first fully connect layer')
    parser.add_argument('--hfeat2', default=300, type=int,
                        help='hidden num of second fully connect layer')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument("--doubleDQN", action="store_true",
                        help="use double DQN")
    parser.add_argument('--policy_update_interval', type=int, default=1, metavar='N',
                        help='Delay policy update if set larger than 1')
    parser.add_argument("--policy_smooth", action="store_true",
                        help="Target policy smoothing regularization")
    parser.add_argument("--priority", action="store_true",
                        help="Priority memory")
    parser.add_argument("--nstep", action="store_true", help="Nstep learning")
    parser.add_argument("--batchnorm", action="store_true",
                        help="batch normalization")
    parser.add_argument("--show", action="store_true",
                        help="show q-estimate vs reward")

    args = parser.parse_args()
    return args

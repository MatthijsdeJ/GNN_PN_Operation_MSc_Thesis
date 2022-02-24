#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:01:28 2022

@author: matthijs
"""

from training.training import Run
import argparse
import auxiliary.util as util
import training.models
from auxiliary.config import config
import copy


def main():
    
    # Specify arguments
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument("-n", "--model_name", help="The name of the model to " +
                        "log to wandb.")
    parser.add_argument("-d", "--processed_tutor_imitation", help="Directory " +
                        "where the subdirectories with data points are " +
                        "stored.")
#    parser.add_argument("-aa","--advanced_analysis", help="Whether to" + 
#                        "perform a more advanced analysis on the val set")
    
    # specify hyperparameter arguments
    parser.add_argument("-mt", "--model_type", help="The model type. Should be GCN or FCNN.", type=str)
    parser.add_argument("-ngl", "--GCN_layers", help="Number of GCN layers in the model", type=int)
    parser.add_argument("-nh", "--N_node_hidden", help="Number of nodes in the hidden layers.", type=int)
    parser.add_argument("-lr", "--lr", help="Learning rate of the optimizer", type=float)
    parser.add_argument("-bs", "--batch_size", help="Batch size. ", type=int)
    parser.add_argument("-wstd", "--weight_init_std", help="Std of the normal from which to sample the weights.",
                        type=float)
    parser.add_argument("-wd", "--weight_decay", help="Weight decay of the optimizer", type=float)
    parser.add_argument("-a", "--aggr", help="Aggregation function of the GCN layers.", type=str)
    parser.add_argument("-nslw", "--non_sub_label_weight", help="Label weights for the objects on the substation not " +
                        "affected in the action.", type=float)   
    parser.add_argument("-lsa", "--label_smoothing_alpha", help="Label smoothing coefficient.", type=float)
    parser.add_argument("-nt", "--network_type", help="Homogeneous or heterogeneous.", type=str)
    parser.add_argument("-nl", "--N_layers", help="Number of hidden layers.", type=int)
    
    # Parse
    args = parser.parse_args()
    
    # If an argument is given, overwrite the config file
    train_parameters = copy.deepcopy(config['training'])

    if args.model_name is not None:
        train_parameters['wandb']['model_name'] = args.model_name
    if args.model_type is not None:
        train_parameters['hyperparams']['model_type'] = args.model_type
    if args.GCN_layers is not None:
        train_parameters['GCN']['hyperparams']['GCN_layers'] = args.GCN_layers
    if args.N_node_hidden is not None:
        train_parameters['hyperparams']['N_node_hidden'] = args.N_node_hidden
    if args.lr is not None:
        train_parameters['hyperparams']['lr'] = args.lr
    if args.batch_size is not None:
        train_parameters['hyperparams']['batch_size'] = args.batch_size
    if args.weight_init_std is not None:
        train_parameters['hyperparams']['weight_init_std'] = args.weight_init_std
    if args.weight_decay is not None:
        train_parameters['hyperparams']['weight_decay'] = args.weight_decay
    if args.aggr is not None:
        train_parameters['GCN']['hyperparams']['aggr'] = args.aggr
    if args.non_sub_label_weight is not None:
        train_parameters['hyperparams']['non_sub_label_weight'] = args.non_sub_label_weight
    if args.label_smoothing_alpha is not None:
        train_parameters['hyperparams']['label_smoothing_alpha'] = args.label_smoothing_alpha
    if args.network_type is not None:
        train_parameters['GCN']['hyperparams']['network_type'] = training.models.GCN.NetworkType(
            args.network_type
        )
    if args.N_layers is not None:
        train_parameters['FCNN']['hyperparams']['N_layers'] = args.N_layers
    # Start the run
    r = Run(train_parameters)
    r.start()


if __name__ == "__main__":
    main()

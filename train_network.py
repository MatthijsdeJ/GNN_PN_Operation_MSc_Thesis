#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:01:28 2022

@author: matthijs
"""

from GCN.training import Run
import argparse
import auxilary.util as util

def main():
    
    #Specify arguments
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument("-n","--model_name", help="The name of the model to " +
                        "log to wandb.")
    parser.add_argument("-d","--processed_tutor_imitation", help="Directory " +
                        "where the subdirectories with data points are " +
                        "stored.")
#    parser.add_argument("-aa","--advanced_analysis", help="Whether to" + 
#                        "perform a more advanced analysis on the val set")
    
    #specify hyperparameter arguments
    parser.add_argument("-nl","--GNN_layers", help="Number of GNN layers " +
                        "in the model", type=int)   
    parser.add_argument("-nh","--N_node_hidden", help="Number of nodes in " +
                        "the hidden layers.", type=int)      
    parser.add_argument("-lr","--lr", help="Learning rate " +
                        "of the optimizer", type=float)   
    parser.add_argument("-bs","--batch_size", help="Batch size. ", type=int)   
    parser.add_argument("-wstd","--weight_init_std", help="Std of the normal" +
                        " from which to sample the weights.", type=float)   
    parser.add_argument("-wd","--weight_decay", help="Weight decay " +
                        "of the optimizer", type=float)   
    parser.add_argument("-a","--aggr", help="Aggregation function " +
                        "of the GNN layers.")   
    parser.add_argument("-nslw","--non_sub_label_weight", help="Label " +
                        "weights for the objects on the substation not " +
                        "affected in the action.", type=float)   
    parser.add_argument("-lsa","--label_smoothing_alpha", help="Label " +
                        "smoothing coefficient.", type=float)   
    parser.add_argument("-nt","--network_type", help="Homogeneous or " +
                        "heterogeneous.")   
    
    #Parse
    args = parser.parse_args()
    
    #If an argument is given, overwrite the config file
    config = util.load_config()
    
    if args.model_name is not None:
        config['training']['wandb']['model_name'] = \
            args.model_name
    if args.processed_tutor_imitation is not None:
        config['paths']['processed_tutor_imitation'] = \
            args.processed_tutor_imitation
    if args.GNN_layers is not None:
        config['training']['hyperparams']['GNN_layers'] = \
            args.GNN_layers
    if args.N_node_hidden is not None:
        config['training']['hyperparams']['N_node_hidden'] = \
            args.N_node_hidden
    if args.lr is not None:
        config['training']['hyperparams']['lr'] = args.lr
    if args.batch_size is not None:
        config['training']['hyperparams']['batch_size'] = \
            args.batch_size
    if args.weight_init_std is not None:
        config['training']['hyperparams']['weight_init_std'] = \
            args.weight_init_std
    if args.weight_decay is not None:
        config['training']['hyperparams']['weight_decay'] = \
            args.weight_decay
    if args.aggr is not None:
        config['training']['hyperparams']['aggr'] = \
            args.aggr
    if args.non_sub_label_weight is not None:
        config['training']['hyperparams']['non_sub_label_weight'] = \
            args.non_sub_label_weight
    if args.label_smoothing_alpha is not None:
        config['training']['hyperparams']['label_smoothing_alpha'] = \
            args.label_smoothing_alpha
    if args.network_type is not None:
        config['training']['hyperparams']['network_type'] = \
            args.network_type
            
    #Start the run
    r = Run(config)
    r.start()

if __name__ == "__main__":
    main()

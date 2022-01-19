#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:01:28 2022

@author: matthijs
"""

from training import Run
import argparse
import util

def main():
    
    #Specify arguments
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument("-n","--model_name", help="The name of the model to " +
                        "log to wandb.")
    parser.add_argument("-d","--processed_tutor_imitation", help="Directory " +
                        "where the subdirectories with data points are " +
                        "stored.")
    parser.add_argument("-nl","--n_gnn_layers", help="Number of GNN layers " +
                        "in the model", type=int)   
    parser.add_argument("-nh","--n_node_hidden", help="Number of nodes in " +
                        "the hidden layers.", type=int)      
    parser.add_argument("-lr","--learing_rate", help="Learning rate " +
                        "of the optimizer", type=float)   
    
    #Parse
    args = parser.parse_args()
    
    #If an argument is given, overwrite the config file
    config = util.load_config()
    config['training']['wandb']['model_name'] = args.model_name
    if args.processed_tutor_imitation is not None:
        config['paths']['processed_tutor_imitation'] = \
            args.processed_tutor_imitation
    if args.n_gnn_layers is not None:
        config['training']['hyperparams']['n_gnn_layers'] = \
            args.n_gnn_layers
    if args.n_node_hidden is not None:
        config['training']['hyperparams']['n_node_hidden'] = \
            args.n_node_hidden
    if args.learing_rate is not None:
        config['training']['hyperparams']['learing_rate'] = \
            args.learing_rate
            
    #Start the run
    r = Run(config)
    r.start()

if __name__ == "__main__":
    main()

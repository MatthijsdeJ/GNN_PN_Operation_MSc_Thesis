#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:30:55 2021

@author: matthijs
"""
import grid2op
import numpy as np
from typing import List
from pathlib import Path

'''
TODO: Functions in this file were created during data exploration, but
have not been tested rigourisly or updated since. Be mindful.
'''

def extract_gen_features(obs_dict: dict) -> np.array:
    '''
    Given the grid2op observation in dictionary form, 
    return the generator features.

    Parameters
    ----------
    obs : dict
        Dictionary epresentation of the grid2op observation. Can be obtained 
        with obs.to_dict().

    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different 
        objects. Colums represent the 'p', 'q', 'v' features.
    '''
    X = np.array(list(obs_dict['gens'].values())).T
    return X

def extract_load_features(obs_dict: dict) -> np.array:
    '''
    Given the grid2op observation in dictionary form, 
    return the generator features.

    Parameters
    ----------
    obs : dict
        Dictionary epresentation of the grid2op observation. Can be obtained 
        with obs.to_dict().

    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different 
        objects. Colums represent the 'p', 'q', 'v' features.
    '''
    X = np.array(list(obs_dict['loads'].values())).T
    return X

def extract_or_features(obs_dict: dict) -> np.array:
    '''
    Given the grid2op observation in dictionary form, 
    return the load features.

    Parameters
    ----------
    obs : dict
        Dictionary epresentation of the grid2op observation. Can be obtained 
        with obs.to_dict().

    Returns
    -------
    X : np.array
        Array representation of the features;  rows correspond to the different 
        objects. Colums represent the 'p', 'q', 'v', 'a', 'line_rho' features.
    '''
    X = np.array(list(obs_dict['lines_or'].values())).T
    X = np.concatenate((X,np.reshape(np.array(obs_dict['rho']),(-1,1))),axis=1)
    return X

def extract_ex_features(obs_dict: dict) -> np.array:
    '''
    Given the grid2op observation in dictionary form, 
    return the generator features.

    Parameters
    ----------
    obs : dict
        Dictionary epresentation of the grid2op observation. Can be obtained 
        with obs.to_dict().
    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different 
        objects. Colums represent the 'p', 'q', 'v', 'a', 'line_rho' features.
    '''
    X = np.array(list(obs_dict['lines_ex'].values())).T
    X = np.concatenate((X,np.reshape(np.array(obs_dict['rho']),(-1,1))),axis=1)
    return X

def get_filepaths(tutor_data_path: str) -> List[Path]:
    '''
    Get the paths of the .npy data files in the directory, with recursive
    effect.

    Parameters
    ----------
    tutor_data_path : str
        String representing the directory path.

    Returns
    -------
    List
        List of the paths of the files.

    '''
    return list(Path(tutor_data_path).rglob('*.npy'))


# =============================================================================
# def extract_features_zero_impunement(obs: grid2op.Observation.CompleteObservation):
#     '''
#     TODO: before use, needs updating, testing, and documentation.
# 
#     Parameters
#     ----------
#     obs : grid2op.Observation.CompleteObservation
#         DESCRIPTION.
# 
#     Returns
#     -------
#     X : TYPE
#         DESCRIPTION.
#     T : TYPE
#         DESCRIPTION.
# 
#     '''
#     N_features = 3+5
#     n=obs.n_gen+obs.n_load+2*obs.n_line
#     
#     X=np.zeros((n,N_features))
#     T=np.zeros(n)
#     for t,f in enumerate([extract_gen_features,extract_load_features,extract_or_features,extract_ex_features]):
#         if t < 2:
#             for i,x in zip(*f(obs)):
#                 X[i,:3]=x
#                 T[i]=t
#         else:
#             for i,x in zip(*f(obs)):
#                 X[i,3:]=x   
#                 T[i]=t
#     return X,T
# =============================================================================


# =============================================================================
# i = obs.gen_pos_topo_vect
# i = obs.load_pos_topo_vect
# i = obs.line_ex_pos_topo_vect
# =============================================================================

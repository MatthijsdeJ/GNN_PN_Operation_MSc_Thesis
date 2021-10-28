#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:30:55 2021

@author: matthijs
"""
import grid2op
import numpy as np


'''
TODO: Functions in this file were created during data exploration, but
have not been tested rigourisly or updated since. Be mindful.
'''

def extract_gen_features(obs: grid2op.Observation.CompleteObservation):
    '''
    TODO: before use, needs updating, testing, and documentation.

    Parameters
    ----------
    obs : grid2op.Observation.CompleteObservation
        DESCRIPTION.

    Returns
    -------
    i : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    '''
    i = obs.gen_pos_topo_vect
    X = np.array(list(obs.to_dict()['gens'].values())).T
    return i,X

def extract_load_features(obs: grid2op.Observation.CompleteObservation):
    '''
    TODO: before use, needs updating, testing, and documentation.

    Parameters
    ----------
    obs : grid2op.Observation.CompleteObservation
        DESCRIPTION.

    Returns
    -------
    i : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    '''
    i = obs.load_pos_topo_vect
    X = np.array(list(obs.to_dict()['loads'].values())).T
    return i,X

def extract_or_features(obs: grid2op.Observation.CompleteObservation):
    '''
    TODO: before use, needs updating, testing, and documentation.

    Parameters
    ----------
    obs : grid2op.Observation.CompleteObservation
        DESCRIPTION.

    Returns
    -------
    i : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    '''
    i = obs.line_or_pos_topo_vect
    X = np.array(list(obs.to_dict()['lines_or'].values())).T
    X = np.concatenate((X,np.reshape(np.array(obs.to_dict()['rho']),(-1,1))),axis=1)
    return i,X

def extract_ex_features(obs: grid2op.Observation.CompleteObservation):
    '''
    TODO: before use, needs updating, testing, and documentation.


    Parameters
    ----------
    obs : grid2op.Observation.CompleteObservation
        DESCRIPTION.

    Returns
    -------
    i : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    '''
    i = obs.line_ex_pos_topo_vect
    X = np.array(list(obs.to_dict()['lines_ex'].values())).T
    X = np.concatenate((X,np.reshape(np.array(obs.to_dict()['rho']),(-1,1))),axis=1)
    return i,X


def extract_features_zero_impunement(obs: grid2op.Observation.CompleteObservation):
    '''
    TODO: before use, needs updating, testing, and documentation.

    Parameters
    ----------
    obs : grid2op.Observation.CompleteObservation
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.

    '''
    N_features = 3+5
    n=obs.n_gen+obs.n_load+2*obs.n_line
    
    X=np.zeros((n,N_features))
    T=np.zeros(n)
    for t,f in enumerate([extract_gen_features,extract_load_features,extract_or_features,extract_ex_features]):
        if t < 2:
            for i,x in zip(*f(obs)):
                X[i,:3]=x
                T[i]=t
        else:
            for i,x in zip(*f(obs)):
                X[i,3:]=x   
                T[i]=t
    return X,T
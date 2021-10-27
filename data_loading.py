#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:23:49 2021

@author: matthijs
"""
from collections import namedtuple
import os
import pickle as pkl
from typing import Union, Tuple
import numpy as np
import grid2op
import util
import Action_space_auto_realistic


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class EpisodeSet():
    """
    Class that represents a group of episodes saved as pickle files.
    Accompanying functions allow access to episodes.
    """
    def __init__(self, location: str):
        '''
        Parameters
        ----------
        location : str
            Path of the folder containing the pickle files containing the pisodes.
        '''
        self.location=location
        self.file_paths=[location +'/' + f for f in os.listdir(location)]
        
        
    def len(self) -> int:
        '''
        Returns the number of episodes.
        '''
        return len(self.file_paths)
    
    def get(self, idx: int) -> Episode:
        '''
        Access an episode by its index.
        '''
        with open(self.file_paths[idx], 'rb') as f:
            episode = pkl.load(f)
        return episode
    
class StepSet():
    '''
    A set of steps in an episode.
    '''
    def __init__(self, episode: Episode):
        '''
        Parameters
        ----------
        episode : Episode
            The episode to extract the steps from.

        '''
        self.steps = episode.steps
            
    def len(self) -> int:
        '''
        Returns the number of steps in the set.
        '''
        return len(self.steps)
    
    def get(self,idx: Union[int, slice]) -> EpisodeStep:
        '''
        Access steps by indexing them.
        '''
        return self.steps[idx]
    
class DataIterator():
    '''
    Class for iterating over the steps over the different episodes.
    Returns the index and EpisodeStep each iteration.
    '''    
    def __init__(self, location: str):
        '''
        Parameters
        ----------
        location : str
            Path of the folder containing the pickle files containing the episodes.
        '''
        self.episode_set = EpisodeSet(location)
        self.i_ep = 0
        self.n_ep = self.episode_set.len()
        self.select_step_set()
        
    def select_step_set(self):
        '''
        Select an episode to iterate over.
        '''
        self.i_step=0
        self.step_set = StepSet(self.episode_set.get(self.i_ep))
        self.n_step = self.step_set.len()
        
    def __next__(self):
        '''
        Obtain the next step.
        
        Raises
        ------
        StopIteration
            There is no other iteration.

        Returns
        -------
        result : Tuple[Tuple[int,int],EpisodeStep]
            The index of step defined by a file index and a step index.
            And the corresponding EpisodeStep.
        '''
        if self.i_step < self.n_step:
            result = (self.i_ep,self.i_step),self.step_set.get(self.i_step)
            self.i_step += 1
        else:
            if self.i_ep < self.n_ep-1:
                self.i_ep += 1
                self.select_step_set()
                result = self.__next__()
            else:
                raise StopIteration
        return result  
    
    def __iter__(self):
        return self
    
def parse_observation(obs: np.array, n_gen: int, n_load: int, n_line: int) \
                        -> Tuple[np.array,np.array,np.array,np.array,np.array,np.array]:
    '''
    Parse the observation array as used in Medha's implementation into seperate parts.

    Parameters
    ----------
    obs : np.array
        The observation array.
    n_gen : int
        The number of generators in the grid.
    n_load : int
        The number of loads in the grid.
    n_line : int
        The number of lines in the grid.

    Returns
    -------
    X_gen : np.array
        The generator features. Columns represent the features:
        P,Q,V
    X_load : np.array
        The load features. Columns represent the features:
        P,Q,V
    X_or : np.array
        The line origin features. Columns represent the features:
        P, Q, V, A
    X_ex : np.array
        The line extremity features. Columns represent the features:
        P, Q, V, A
    X_line : np.array 
        The line features. Columns represent the features:
        rho, line status, timestep since overflow
    topo_vect : np.array
        The topology vector indicating the connections of objects to busbars.
        Each index represents one object; a 0 value indicates no connection,
        a 1 a connection to the first busbar, a 2 a connection to the second busbar.
    '''
    f_gen,f_load,f_endpoint,f_line = 3,3,4,3 #Numbers of features

    i=0
    X = []
    for n,f in zip([n_gen,n_load,n_line,n_line,n_line],
                        [f_gen,f_load,f_endpoint,f_endpoint,f_line]):     
        X.append(obs[i:i+n*f].reshape((f,n)).T)
        i+=n*f
    
    X_gen, X_load, X_or, X_ex, X_line = X
    topo_vect = obs[i:]
    
    return X_gen, X_load, X_or, X_ex, X_line, topo_vect

def preprocess_observation(obs: np.array, obs_space: grid2op.Observation.ObservationSpace)\
                            -> Tuple[np.array,np.array,np.array,np.array,np.array,np.array]:
    '''
    Preprocess the observation of a timestep into the features for the generators, loads, 
    line origins, and line extremities, as well as the connectivity matrix.

    Parameters
    ----------
    obs : np.array
        The observation array.
    obs_space : grid2op.Observation.ObservationSpace
        The observation space representing the environment.

    Returns
    -------
    X_gen : np.array
        The generator features. Columns represent the features:
        P,Q,V
    X_load : np.array
        The load features. Columns represent the features:
        P,Q,V
    X_or : np.array
        The line origin features. Columns represent the features:
        P, Q, V, A, line rho, line status, line timestep since overflow
    X_ex : np.array
        The line extremity features. Columns represent the features:
        P, Q, V, A, line rho, line status, line timestep since overflow
    topo_vect : np.array
        The topology vector indicating the connections of objects to busbars.
        Each index represents one object; a 0 value indicates no connection,
        a 1 a connection to the first busbar, a 2 a connection to the second busbar.
    connectivity_edges : np.array
        Representation of which objects are connected by edges. Array rows
        represent indices in the adjacency matrix.
    '''
    
    X_gen, X_load, X_or, X_ex, X_line, topo_vect = parse_observation(obs, obs_space.n_gen, 
                                                                     obs_space.n_load, obs_space.n_line)
    X_or = np.concatenate((X_or,X_line),axis=1) #Include the line features with the line origin
    X_ex = np.concatenate((X_ex,X_line),axis=1) #Include the line features with the line extremity

    connectivity_edges = util.connectivity_matrix(obs_space.sub_info, topo_vect, X_line[:,1], 
                                              obs_space.line_or_pos_topo_vect, obs_space.line_ex_pos_topo_vect, 
                                              obs_space.dim_topo).T
    
    return X_gen, X_load, X_or, X_ex, topo_vect, connectivity_edges

class action_identificator():
    '''
    Class for identifying action IDs as originating from Medha's model and
    retrieving the corresponding Grid2Op actions. The actions are limited
    to instances of setting the topology vector.
    
    A class to reduce overhead.
    '''
    
    def __init__(self):
        self.all_actions,self.DN_actions = \
                Action_space_auto_realistic.get_env_actions()
        
    def get_set_topo_vect(self, action_id: int):
        '''
        Retrieve the 'set_topo_vect' attribute containing the set
        object-busbar connections belonging, identified by a particular id.

        Parameters
        ----------
        action_id : int
            The id of the action.

        Returns
        -------
        np.array
            The array indicating with object-busbar connections were set.
            A 0 represent no change, 1 set to the first busbar, 2 set to the second busbar.

        '''
        return self.all_actions[action_id]._set_topo_vect
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:27:46 2022

@author: matthijs
"""
import numpy as np
from typing import Sequence, Tuple, List, Optional
import grid2op
from grid2op.dtypes import dt_int
import math


def extract_gen_features(obs_dict: dict) -> np.array:
    """
    Given the grid2op observation in dictionary form,
    return the generator features.

    Parameters
    ----------
    obs_dict : dict
        Dictionary representation of the grid2op observation. Can be obtained
        with obs.to_dict().

    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different
        objects. Columns represent the 'p', 'q', 'v' features.
    """
    X = np.array(list(obs_dict['gens'].values())).T
    return X


def extract_load_features(obs_dict: dict) -> np.array:
    """
    Given the grid2op observation in dictionary form,
    return the generator features.

    Parameters
    ----------
    obs_dict : dict
        Dictionary representation of the grid2op observation. Can be obtained
        with obs.to_dict().

    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different
        objects. Columns represent the 'p', 'q', 'v' features.
    """
    X = np.array(list(obs_dict['loads'].values())).T
    return X


def extract_or_features(obs_dict: dict, thermal_limits: Sequence[int]) \
                        -> np.array:
    """
    Given the grid2op observation in dictionary form,
    return the load features.

    Parameters
    ----------
    obs_dict : dict
        Dictionary representation of the grid2op observation. Can be obtained
        with obs.to_dict().
    thermal_limits : Sequence[int]
        Sequence with the thermal limits of the lines.

    Returns
    -------
    X : np.array
        Array representation of the features;  rows correspond to the different
        objects. Columns represent the 'p', 'q', 'v', 'a', 'line_rho',
        'line_capacity' features.
    """
    X = np.array(list(obs_dict['lines_or'].values())).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.concatenate((X,
                            np.reshape(np.array(obs_dict['rho']), (-1, 1)),
                            np.reshape(np.array(thermal_limits), (-1, 1))),
                           axis=1)
    return X


def extract_ex_features(obs_dict: dict, thermal_limits: Sequence[int]) \
                        -> np.array:
    """
    Given the grid2op observation in dictionary form,
    return the generator features.

    Parameters
    ----------
    obs_dict : dict
        Dictionary representation of the grid2op observation. Can be obtained
        with obs.to_dict().
    thermal_limits : Sequence[int]
        Sequence with the thermal limits of the lines.
    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different
        objects. Columns represent the 'p', 'q', 'v', 'a', 'line_rho',
        'line_capacity' features.
    """
    X = np.array(list(obs_dict['lines_ex'].values())).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.concatenate((X,
                            np.reshape(np.array(obs_dict['rho']), (-1, 1)),
                            np.reshape(np.array(thermal_limits), (-1, 1))),
                           axis=1)
    return X


def connectivity_matrices(sub_info: Sequence[int], 
                          topo_vect: Sequence[int], 
                          line_or_pos_topo_vect: Sequence[int], 
                          line_ex_pos_topo_vect: Sequence[int]
                          ) -> Tuple[np.array, np.array, np.array]:
    """
    Computes and return three connectivity matrices, based on three possible
    relations between objects. Matrices are returned as sparse matrices,
    represented by the indices of the edges. All relations are bidirectional,
    i.e. duplicated.

    Parameters
    ----------
    sub_info : Sequence[int]
        The number of objects per substation.
    topo_vect : Sequence[int]
        The bus to which each object is connected.
    line_or_pos_topo_vect : Sequence[int]
        The indices in the topo vector of the line origins.
    line_ex_pos_topo_vect : Sequence[int]
        The indices in the topo vector of the line extremities.


    Returns
    -------
    connectivity_matrix_samebus: np.array
        The sparse connectivity matrix between objects connected to the same bus
        of their substation.
    connectivity_matrix_otherbus = np.array
        The sparse connectivity matrix between objects connected to the other bus
        of their substation.
    connectivity_matrix_line = np.array
        The sparse connectivity matrix between objects connected by lines.
    """

    beg_ = 0
    end_ = 0
    row_ind_samebus = []
    col_ind_samebus = []
    row_ind_otherbus = []
    col_ind_otherbus = []
    row_ind_line = []
    col_ind_line = []         
    
    for sub_id, nb_obj in enumerate(sub_info):
        # it must be a vanilla python integer, otherwise it's not handled by 
        # some backend
        # especially if written in c++
        nb_obj = int(nb_obj)
        end_ += nb_obj
        # tmp = np.zeros(shape=(nb_obj, nb_obj), dtype=dt_float)
        for obj1 in range(nb_obj):
            my_bus = topo_vect[beg_+obj1]
            if my_bus == -1:
                # object is disconnected, nothing is done
                continue
            # connect an object to itself
#                 row_ind.append(beg_ + obj1)
#                 col_ind.append(beg_ + obj1)
#                 WHY??

            # connect the other objects to it
            for obj2 in range(obj1+1, nb_obj):
                my_bus2 = topo_vect[beg_+obj2]
                if my_bus2 == -1:
                    # object is disconnected, nothing is done
                    continue
                if my_bus == my_bus2:
                    # objects are on the same bus
                    # tmp[obj1, obj2] = 1
                    # tmp[obj2, obj1] = 1
                    row_ind_samebus.append(beg_ + obj2)
                    col_ind_samebus.append(beg_ + obj1)
                    row_ind_samebus.append(beg_ + obj1)
                    col_ind_samebus.append(beg_ + obj2)
                else:
                    # objects are on different bus 
                    row_ind_otherbus.append(beg_ + obj2)
                    col_ind_otherbus.append(beg_ + obj1)
                    row_ind_otherbus.append(beg_ + obj1)
                    col_ind_otherbus.append(beg_ + obj2)                      
        beg_ += nb_obj

    # both ends of a line are connected together (if line is connected)
    for q_id in range(len(line_or_pos_topo_vect)):
        if topo_vect[line_or_pos_topo_vect][q_id] != -1:
            # if powerline is connected, connect both its side
            row_ind_line.append(line_or_pos_topo_vect[q_id])
            col_ind_line.append(line_ex_pos_topo_vect[q_id])
            row_ind_line.append(line_ex_pos_topo_vect[q_id])
            col_ind_line.append(line_or_pos_topo_vect[q_id])
         
    row_ind_samebus = np.array(row_ind_samebus).astype(dt_int)
    col_ind_samebus = np.array(col_ind_samebus).astype(dt_int)
    row_ind_otherbus = np.array(row_ind_otherbus).astype(dt_int)
    col_ind_otherbus = np.array(col_ind_otherbus).astype(dt_int)
    row_ind_line = np.array(row_ind_line).astype(dt_int)
    col_ind_line = np.array(col_ind_line).astype(dt_int)
           
    connectivity_matrix_samebus = np.stack((row_ind_samebus, col_ind_samebus))
    connectivity_matrix_otherbus = np.stack((row_ind_otherbus, col_ind_otherbus))
    connectivity_matrix_line = np.stack((row_ind_line, col_ind_line))

    return connectivity_matrix_samebus, \
        connectivity_matrix_otherbus, \
        connectivity_matrix_line


def tv_groupby_subst(tv: Sequence, sub_info: Sequence[int]) -> \
        List[Sequence]:
    """
    Group a sequence the shape of the topology vector by the substations.

    Parameters
    ----------
    tv : Sequence
        Sequence the shape of the topology vector.
    sub_info : Sequence[int]
        Sequence with elements containing the number of object connected to
        each substation.

    Returns
    -------
    List[Sequence]
        List, each element corresponding to a Sequence of objects in tv that
        belong to a particular substation.
    """
    i = 0
    gs = []
    for ss in sub_info:
        gs.append(tv[i:i+ss])
        i += ss
    return gs


def init_env(config: dict,
             gamerules_class: grid2op.Rules.BaseRules,
             ) -> grid2op.Environment.Environment:
    """
    Prepares the Grid2Op environment from a dictionary containing configuration
    setting.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration variables.
    gamerules_class : grid2op.Rules.BaseRules
        The rules of the game.

    Returns
    -------
    env : TYPE
        The Grid2Op environment.
    """
    data_path = config['paths']['rte_case14_realistic']
    scenario_path = config['paths']['rte_case14_realistic_chronics']

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=data_path, 
                           chronics_path=scenario_path, 
                           backend=backend,
                           gamerules_class=gamerules_class,
                           test=True)
    except ImportError:
        env = grid2op.make(dataset=data_path, 
                           chronics_path=scenario_path,
                           gamerules_class=gamerules_class,
                           test=True)
        
    # for reproducible experiments
    env.seed(config['tutor_generated_data']['seed'])  

    # Set custom thermal limits
    thermal_limits = config['rte_case14_realistic']['thermal_limits']
    env.set_thermal_limit(thermal_limits)
    
    return env


def ts_to_day(ts: int, ts_in_day: int) -> int:
    """
    Calculate what day (as a number) a timestep is in.

    Parameters
    ----------
    ts : int
        The timestep.
    ts_in_day : int
        The number of timesteps in a day.

    Returns
    -------
    int
        The day.
    """
    return math.floor(ts/ts_in_day)


def skip_to_next_day(env: grid2op.Environment.Environment,
                     ts_in_day: int,
                     chronic_id: int,
                     disable_line: int) -> Optional[dict]:
    """
    Skip the environment to the next day.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The environment to fast-forward to the next day in.
    ts_in_day : int
        The number of timesteps in a day.
    chronic_id : int
        The current chronic id.
    disable_line : int
        The index of the line to be disabled.

    Returns
    -------
    info : dict
        Grid2op dict given out as the fourth output of env.step(). Contains
        the info about whether an error has occurred.
    """

    ts_next_day = ts_in_day*(1+ts_to_day(env.nb_time_step,
                                         ts_in_day))
    env.set_id(chronic_id)
    _ = env.reset()
    
    if disable_line != -1:
        env.fast_forward_chronics(ts_next_day-1)
        _, _, _, info = env.step(env.action_space(
            {"set_line_status": (disable_line, -1)}))
        return info
    else:
        env.fast_forward_chronics(ts_next_day)
        return None

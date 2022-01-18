#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:04:42 2021

@author: matthijs
"""
import numpy as np
from grid2op.dtypes import dt_int, dt_float, dt_bool
import yaml
import pkg_resources
import os
from typing import List, Tuple, Sequence

def connectivity_matrices(sub_info: Sequence[int], 
                          topo_vect: Sequence[int], 
                          line_or_pos_topo_vect: Sequence[int], 
                          line_ex_pos_topo_vect: Sequence[int]
                          ) -> Tuple[np.array,np.array,np.array]:
    '''
    Computes and return three connectivity matrices, based on three possible relations between objects.
    Matrices are returned as sparse matrices, represented by the indices of the edges.
    All relations are bidirectional, i.e. duplicated.

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
    '''

    beg_ = 0
    end_ = 0
    row_ind_samebus = []
    col_ind_samebus = []
    row_ind_otherbus = []
    col_ind_otherbus = []
    row_ind_line = []
    col_ind_line = []         
    
    for sub_id, nb_obj in enumerate(sub_info):
        # it must be a vanilla python integer, otherwise it's not handled by some backend
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
        if topo_vect[line_or_pos_topo_vect][q_id]!=-1:
            # if powerline is connected connect both its side
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
           
    connectivity_matrix_samebus = np.stack((row_ind_samebus,col_ind_samebus))
    connectivity_matrix_otherbus = np.stack((row_ind_otherbus,col_ind_otherbus))
    connectivity_matrix_line = np.stack((row_ind_line,col_ind_line))

    return connectivity_matrix_samebus,  connectivity_matrix_otherbus, connectivity_matrix_line
    
def load_config():
    '''
    Loads the config file as dictionary

    Raises
    ------
    exc
        YAML exception encountered by the YAML parsers.

    Returns
    -------
    dict
        The config file.

    '''
    with open('config.yaml') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
        
def set_wd_to_package_root():
    '''
    Set the working directory to the root of the package.
    '''
    os.chdir(os.path.dirname(__file__))
    
def flatten(t: Sequence[Sequence]) -> List:
    '''
    Flatten a sequence of sequences.

    Parameters
    ----------
    t : Sequence[Sequence]
        The sequence of sequences to flatten.

    Returns
    -------
    List
        Flattened sequence.
    '''
    return [item for sublist in t for item in sublist]

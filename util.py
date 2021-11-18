#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:04:42 2021

@author: matthijs
"""
import numpy as np
from grid2op.dtypes import dt_int, dt_float, dt_bool
import yaml

def connectivity_matrix(sub_info, topo_vect, line_status, line_or_pos_topo_vect, 
                        line_ex_pos_topo_vect, dim_topo, as_edge_indices=True):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        Let "dim_topo := 2 * n_line + n_prod + n_conso + n_storage" (the total number of elements on the grid)

        It is a matrix of size dim_topo, dim_topo, with values 0 or 1.
        For two objects (lines extremity, generator unit, load) i,j :

            - if i and j are connected on the same substation:
                - if `conn_mat[i,j] = 0` it means the objects id'ed i and j are not connected to the same bus.
                - if `conn_mat[i,j] = 1` it means the objects id'ed i and j are connected to the same bus

            - if i and j are not connected on the same substation then`conn_mat[i,j] = 0` except if i and j are
              the two extremities of the same power line, in this case `conn_mat[i,j] = 1` (if the powerline is
              in service or 0 otherwise).

        Returns
        -------
        res: ``numpy.ndarray``, shape:dim_topo,dim_topo, dtype:float
            The connectivity matrix, as defined above

        Notes
        -------
        Matrix can be either a sparse matrix or a dense matrix depending on the argument `as_csr_matrix`
        """
        beg_ = 0
        end_ = 0
        row_ind = []
        col_ind = []
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
                        row_ind.append(beg_ + obj2)
                        col_ind.append(beg_ + obj1)
                        row_ind.append(beg_ + obj1)
                        col_ind.append(beg_ + obj2)
            beg_ += nb_obj

        # both ends of a line are connected together (if line is connected)
        for q_id in range(len(line_status)):
            if line_status[q_id]:
                # if powerline is connected connect both its side
                row_ind.append(line_or_pos_topo_vect[q_id])
                col_ind.append(line_ex_pos_topo_vect[q_id])
                row_ind.append(line_ex_pos_topo_vect[q_id])
                col_ind.append(line_or_pos_topo_vect[q_id])
        row_ind = np.array(row_ind).astype(dt_int)
        col_ind = np.array(col_ind).astype(dt_int)
        if not as_edge_indices:
            _connectivity_matrix_ = np.zeros(shape=(dim_topo, dim_topo), dtype=dt_float)
            _connectivity_matrix_[row_ind.T, col_ind] = 1.0
        else:
            _connectivity_matrix_ = np.stack((row_ind,col_ind))
#             data = np.ones(row_ind.shape[0], dtype=dt_float)
#             _connectivity_matrix_ = csr_matrix((data, (row_ind, col_ind)),
#                                                     shape=(dim_topo, dim_topo),
#                                                     dtype=dt_float)
        return _connectivity_matrix_
    
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
    
    with open("config.yaml") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
        
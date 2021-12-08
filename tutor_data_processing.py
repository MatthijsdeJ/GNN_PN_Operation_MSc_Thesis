#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:30:55 2021

@author: matthijs
"""
import grid2op
import numpy as np
from typing import List, Tuple, Callable
from pathlib import Path, PosixPath
import re
import util
import json

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

def get_filepaths(tutor_data_path: str) -> List[PosixPath]:
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

def extract_data_from_filepath(relat_fp: PosixPath) \
                        -> Tuple[int, float, int, int]:
    '''
    Given a relative filepath, extract the information contained in this
    filepath.

    Parameters
    ----------
    relat_fp : PosixPath
        The relative filepath.

    Returns
    -------
    Tuple[int, float, int, int]
        Tuple containing the values of the index of the disabled line,
        the threshold at which no actions were taken, the id of the chronic,
        and the number of days completed.
    '''
    regex_str = 'records_chronics_lout:(.*)_dnthreshold:(.*)' + \
        '/records_chronic:(.*)_dayscomp:(.*).npy'
    line_disabled, dn_threshold, chronic_id, dayscomp = \
                    re.search(regex_str, str(relat_fp)).groups()
    return int(line_disabled), float(dn_threshold), int(chronic_id), \
            int(dayscomp)

def extract_data_from_single_ts(ts_vect: np.array, grid2op_vect_len: int,
                                vect2obs_func: Callable) -> dict:
    '''
    Given the vector of a datapoint rperesenting a single timestep, extract
    the interesting data from this vector and return it as a dictionary.

    Parameters
    ----------
    ts_vect : np.array
        The vector.
    grid2op_vect_len : int
        The length of the vector that represents a grid2op observation.
    vect2obs_func : Callable
        Function for transferring a vector represention of a grid2op 
        observation to the corresponding grid2op observation object.

    Returns
    -------
    dict
        The dictionary containing the relevant data.
    '''
    grid2op_obs_vect = ts_vect[-grid2op_vect_len:]
    obs = vect2obs_func(grid2op_obs_vect)
    obs_dict = obs.to_dict()

    data = {'action_index': int(ts_vect[0]),
            'timestep': int(ts_vect[4]),
            'gen_features': extract_gen_features(obs_dict),
            'load_features': extract_load_features(obs_dict),
            'or_features': extract_or_features(obs_dict),
            'ex_features': extract_ex_features(obs_dict),
            'topo_vect': obs_dict['topo_vect'].copy()
           }
    return data

def hash_nparray(arr: np.array) -> int:
    '''
    Hashes a numpy array.

    Parameters
    ----------
    arr : np.array
        The array.

    Returns
    -------
    int
        The hash value.
    '''
    return hash(arr.data.tobytes())

class NumpyEncoder(json.JSONEncoder):
    '''
    Class that can be used in json.dump() to encode np.array objects.
    '''
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class con_matrix_cache():
    '''
    Connectivity matrices are expensive to compute and store and many 
    datapoints might share the same connectivity matrix. For this reason, we
    only compute/store each con. matrix once, and instead provide data points
    with a hash pointing to the correct con. matrix.
    '''
    
    def __init__(self):
        self.con_matrices = {}
        
    def get_key_add_to_dict(self, topo_vect: np.array, 
                            sub_info: np.array,
                            line_or_pos_topo_vect: np.array,
                            line_ex_pos_topo_vect: np.array
                            ) -> int:
        '''
        This function fulfils two purposes: (1) if the corresponding con. 
        matrix hasn't been stored yet, compute it and store it; 
        (2) return the hash key of the corresponding con. matrix.

        Parameters
        ----------
        topo_vect : np.array
            The topology vector from which to compute the con. matrix.
        sub_info : np.array
            Vector representing the number of objects per substation.
        line_or_pos_topo_vect : np.array
            Vector representing the indices of the line origins in the topo 
            vect.
        line_ex_pos_topo_vect : np.array
            Vector representing the indices of the line extremities in the topo 
            vect.
            
        Returns
        -------
        h_topo_vect : int
            The hash of the topology vector, which can be used to index
            for con. matrices.

        '''
        h_topo_vect = hash_nparray(topo_vect)
        if h_topo_vect not in self.con_matrices:
            con_matrix = util.connectivity_matrix(sub_info.astype(int),
                                                  topo_vect.astype(int),
                                        line_or_pos_topo_vect.astype(int),
                                        line_ex_pos_topo_vect.astype(int),
                                        sum(sub_info))
            self.con_matrices[h_topo_vect] = (topo_vect,con_matrix)

        return h_topo_vect
    
    def save(self,fpath:str =''):
        '''
        Save the dictionary of connectivity matrices as a json file.

        Parameters
        ----------
        fpath : str, optional
            Where to store the json file. The default is the wdir.

        '''
        with open(fpath+'con_matrices.json', 'w') as outfile:
            json.dump(self.con_matrices, outfile, cls=NumpyEncoder)
            
    
def save_data_to_file(data: List[dict], output_data_path: str):
    '''
    Given a list of dictionaries, representing various data points,
    save these to a json file. If the list is empty, save nothing.
    
    Parameters
    ----------
    data : List[dict]
        Various data points.
    output_data_path : str
        The output directory where to save the file.
    '''
    if not data:
        return
    
    filename = f'data_lout{data[0]["line_disabled"]}_' + \
        f'chr{data[0]["chronic_id"]}.json'
    with open(output_data_path + filename, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)
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

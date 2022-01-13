#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:30:55 2021

@author: matthijs
"""
import grid2op
import numpy as np
from typing import List, Tuple, Callable, Sequence
from pathlib import Path, PosixPath
import re
import util
import json
import collections

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

def extract_or_features(obs_dict: dict, thermal_limits: Sequence[int]) -> np.array:
    '''
    Given the grid2op observation in dictionary form, 
    return the load features.

    Parameters
    ----------
    obs : dict
        Dictionary epresentation of the grid2op observation. Can be obtained 
        with obs.to_dict().
    thermal_limits : Sequence[int]
        Sequence with the thermal limits of the lines.
        
    Returns
    -------
    X : np.array
        Array representation of the features;  rows correspond to the different 
        objects. Colums represent the 'p', 'q', 'v', 'a', 'line_rho', 
        'line_capacity' features.
    '''
    X = np.array(list(obs_dict['lines_or'].values())).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.concatenate((X,
                            np.reshape(np.array(obs_dict['rho']),(-1,1)),
                            np.reshape(np.array(thermal_limits),(-1,1))),
                         axis=1)
    return X

def extract_ex_features(obs_dict: dict, thermal_limits: Sequence[int]) -> np.array:
    '''
    Given the grid2op observation in dictionary form, 
    return the generator features.

    Parameters
    ----------
    obs : dict
        Dictionary epresentation of the grid2op observation. Can be obtained 
        with obs.to_dict().
    thermal_limits : Sequence[int]
        Sequence with the thermal limits of the lines.
    Returns
    -------
    X : np.array
        Array representation of the features; rows correspond to the different 
        objects. Colums represent the 'p', 'q', 'v', 'a', 'line_rho', 
        'line_capacity' features.
    '''
    X = np.array(list(obs_dict['lines_ex'].values())).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.concatenate((X,
                            np.reshape(np.array(obs_dict['rho']),(-1,1)),
                            np.reshape(np.array(thermal_limits),(-1,1))),
                           axis=1)
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
                                vect2obs_func: Callable, line_disabled: int,
                                env_info_dict: dict, thermal_limits: Sequence[int]) -> dict:
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
    line_disabled : int
        The line index to be disabled. -1 if no line is disabled.     
    env_info_dict: dict
        Dictionary with variables from the environment. Important here,
        the index in the topo vect of the disabled line origin/extremity.
    thermal_limits : Sequence[int]
        Sequence with the thermal limits of the lines.
        
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
            'or_features': extract_or_features(obs_dict, thermal_limits),
            'ex_features': extract_ex_features(obs_dict, thermal_limits),
            'topo_vect': obs_dict['topo_vect'].copy()
           }
    

    #Remove the disabled line from the data, if necessary
    if line_disabled != -1:
        data['or_features'] = np.delete(data['or_features'],line_disabled,axis=0)
        data['ex_features'] = np.delete(data['ex_features'],line_disabled,axis=0)
        data['topo_vect'] = np.delete(data['topo_vect'],[
                                        env_info_dict['dis_line_or_tv'],
                                        env_info_dict['dis_line_ex_tv']])
         
    #Assert the topo_vect has the same length as the features
    assert len(data['topo_vect']) == len(data['gen_features']) + \
                                     len(data['load_features']) + \
                                     len(data['or_features']) + \
                                     len(data['ex_features'])
    return data

def env_info_line_disabled(env:  grid2op.Environment.Environment, 
                           line_disabled: int) -> dict:
    '''
    Generates the adapted grid2op environment variables for the possible 
    disablement of a line. This essentially removes the corresponding
    origin and extremity from the variables.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The grid2op environment.
    line_disabled : int
        The line index to be disabled. -1 if no line is disabled.

    Returns
    -------
    dict
        The dictionary with the information. Entries:
            'sub_info': number of elements per substation
            'gen_pos_topo_vect': indices in the topo vect for each generator
                                 object
            'load_pos_topo_vect': indices in the topo vect for each load
                                 object
            'line_or_pos_topo_vect': indices in the topo vect for each origin
                                 object          
            'line_ex_pos_topo_vect': indices in the topo vect for each extremity
                                 object 
        POSSIBLY:
            'dis_line_or_tv': index in the topo vect of the disabled line origin
            'dis_line_ex_tv': index in the topo vect of the disabled line extremity
    '''
    sub_info = env.sub_info.copy()
    gen_pos_topo_vect = env.gen_pos_topo_vect.copy()
    load_pos_topo_vect = env.load_pos_topo_vect.copy()
    line_or_pos_topo_vect = env.line_or_pos_topo_vect.copy()
    line_ex_pos_topo_vect = env.line_ex_pos_topo_vect.copy()
    
    if line_disabled != -1:
        dis_line_or_tv = line_or_pos_topo_vect[line_disabled]
        dis_line_ex_tv = line_ex_pos_topo_vect[line_disabled]
                                        
        #Remove line at index from line_or/ex_pos_topo_vect
        line_or_pos_topo_vect = np.delete(line_or_pos_topo_vect,line_disabled)
        line_ex_pos_topo_vect = np.delete(line_ex_pos_topo_vect,line_disabled)
        
        #Lowering numbers in the sub_info array
        sub_info[env.line_or_to_subid[line_disabled]] -=1
        sub_info[env.line_ex_to_subid[line_disabled]] -=1
        
        #Lowering indices in the rest of the arrays indexing the topo_vect
        gen_pos_topo_vect = np.array([i - (i>dis_line_or_tv) - \
                                (i>dis_line_ex_tv) for i in gen_pos_topo_vect])
        load_pos_topo_vect = np.array([i - (i>dis_line_or_tv) - \
                                (i>dis_line_ex_tv) for i in load_pos_topo_vect])
        line_or_pos_topo_vect = np.array([i - (i>dis_line_or_tv) - \
                                (i>dis_line_ex_tv) for i in line_or_pos_topo_vect])
        line_ex_pos_topo_vect = np.array([i - (i>dis_line_or_tv) - \
                                (i>dis_line_ex_tv) for i in line_ex_pos_topo_vect])
     

    concat_ptvs = np.concatenate([gen_pos_topo_vect,load_pos_topo_vect, 
                                  line_or_pos_topo_vect,line_ex_pos_topo_vect])
    #Check that the arrays indexing the topo vect are disjoint 
    assert len(set(concat_ptvs)) == len(gen_pos_topo_vect) + len(load_pos_topo_vect) + \
        len(line_or_pos_topo_vect) + len(line_ex_pos_topo_vect)
    #Check that the sub_info max. index (plus one) equals the nr. of indices 
    #equals the sum of objects
    assert max(concat_ptvs)+1 == len(concat_ptvs) == sum(sub_info)
               
    info_dict= {'sub_info':sub_info,
           'gen_pos_topo_vect': gen_pos_topo_vect,
           'load_pos_topo_vect': load_pos_topo_vect,
           'line_or_pos_topo_vect': line_or_pos_topo_vect,
           'line_ex_pos_topo_vect': line_ex_pos_topo_vect}
    if line_disabled != -1:
        info_dict['dis_line_or_tv'] = dis_line_or_tv
        info_dict['dis_line_ex_tv'] = dis_line_ex_tv
        
    return info_dict

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
    
class ConMatrixCache():
    '''
    Connectivity matrices are expensive to compute and store and many 
    datapoints might share the same connectivity matrix. For this reason, we
    only compute/store each con. matrix once, and instead provide data points
    with a hash pointing to the correct con. matrix.
    '''
    
    def __init__(self):
        self.con_matrices = {}
        
    def get_key_add_to_dict(self, topo_vect: np.array, 
                            line_disabled: int,
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
        line_disabled : int
            The line index to be disabled. -1 if no line is disabled.     
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
        #Check that line_or_pos_topo_vect and line_ex_pos_topo_vect 
        #have no overlap
        assert set(line_or_pos_topo_vect).isdisjoint(set(line_ex_pos_topo_vect))
        #And have the same size
        assert len(line_or_pos_topo_vect)==len(line_ex_pos_topo_vect)
        #Check that the number of objcets according to sub_info and topo_vect
        #are equal
        assert sum(sub_info)==len(topo_vect)
        
        h_topo_vect = hash((line_disabled,hash_nparray(topo_vect)))
        if h_topo_vect not in self.con_matrices:
            
            con_matrices = util.connectivity_matrices(sub_info.astype(int),
                                                  topo_vect.astype(int),
                                        line_or_pos_topo_vect.astype(int),
                                        line_ex_pos_topo_vect.astype(int))

            self.con_matrices[h_topo_vect] = (topo_vect,con_matrices)


        return h_topo_vect
    
    def save(self,fpath:str =''):
        '''
        Save the dictionary of connectivity matrices as a json file.

        Parameters
        ----------
        fpath : str, optional
            Where to store the json file. The default is the wdir.

        '''
        with open(fpath, 'w') as outfile:
            json.dump(self.con_matrices, outfile, cls=NumpyEncoder)
            
    @classmethod
    def load(cls, fpath: str):
        '''
        Factory class: initialize a ConMatrixCache based on a file.

        Parameters
        ----------
        fpath : str
            The filepath of the file.
        '''
        cmc = cls()
        with open(fpath, 'r') as file:
            cmc.con_matrices = json.loads(file.read())
        return cmc
            
    
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
        
class FeatureStatistics():
    '''
    Used to track the statistics about features (N, mean, std), which are used 
    in feature normalization.
    
    Since the dataset is too large to hold in memory completely, the feature 
    statistics are computed iteratively.
    '''
    def __init__(self):   
        #Initialize statistics about features
        self.N_gen, self.N_load, self.N_line = 0,0,0
        self.S_gen, self.S_load, self.S_or, self.S_ex = None,None,None,None
        self.S2_gen, self.S2_load, self.S2_or, self.S2_ex = None,None,None,None
    
    def update_feature_statistics(self,data: dict):
        '''
        Update the statistics (number, sum, sum of squares) of the feature
        values.

        Parameters
        ----------
        data : np.array
            Dictionary representing the datapoints, containing the features.

        '''
        features = [data['gen_features'],data['load_features'],
                    data['or_features'],data['ex_features']]
        
        #Update number of objects
        self.N_gen, self.N_load, self.N_line = [n+f.shape[0] for f,n in 
                    zip(features[:-1],[self.N_gen, self.N_load, self.N_line])]
        
        if self.S_gen is None:
            #Initialize sum
            self.S_gen, self.S_load, self.S_or, self.S_ex = \
                        [f.sum(axis=0) for f in features]
            #Initialize sum of squares
            self.S2_gen, self.S2_load, self.S2_or, self.S2_ex = \
                        [(f**2).sum(axis=0) for f in features]
        else:
            #Increase the sum
            self.S_gen, self.S_load, self.S_or, self.S_ex = \
                [s+f.sum(axis=0) for f,s in zip(features,
                [self.S_gen, self.S_load, self.S_or, self.S_ex])]
            #Increase the sum of squares
            self.S2_gen, self.S2_load, self.S2_or, self.S2_ex = \
                [s2+(f**2).sum(axis=0) for f,s2 in zip(features,
                [self.S2_gen, self.S2_load, self.S2_or, self.S2_ex])]
                
    def save_feature_statistics(self, fpath: str) -> dict:
        '''
        Save the feature statistics in the form of the mean and standard 
        deviation per obejct type to a specified location.
        
        Parameters
        ----------
        fpath : str
            The filepath to save to.
        '''
        def std(N,S,S2):
            return np.sqrt(S2/N-(S/N)**2) 
        
        stats = {}
        for name, N, S, S2 in [('gen',self.N_gen,self.S_gen,self.S2_gen),
                                ('load',self.N_load,self.S_load,self.S2_load),
                                ('or',self.N_line,self.S_or,self.S2_or),
                                ('ex',self.N_line,self.S_ex,self.S2_ex)]:
            stats[name] = {'mean':S/N,
                              'std':std(N,S,S2)}
        with open(fpath, 'w') as outfile:
            json.dump(stats, outfile, cls=NumpyEncoder)
    
# =============================================================================
#     def update_datapoint(self, data):
#         self.N +=1
#         self.action_hash_counter[data['']] +=1 
#         self.topo_vect_hash_counter[data['']] +=1 
#         self.timestep_counter[data['timestep']] +=1 
# =============================================================================
            

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

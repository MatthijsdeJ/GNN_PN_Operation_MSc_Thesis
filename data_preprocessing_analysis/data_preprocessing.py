#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:30:55 2021

@author: matthijs
"""
import random

import grid2op
import numpy as np
from typing import List, Tuple, Callable, Sequence
from pathlib import Path, PosixPath
import re
import json
import auxiliary.grid2op_util as g2o_util
import auxiliary.util as util
from auxiliary.util import NumpyEncoder
from auxiliary.config import get_config
from tqdm import tqdm
from auxiliary.generate_action_space import action_identificator
import os
import shutil


def get_filepaths(tutor_data_path: str) -> List[Path]:
    """
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

    """
    return list(Path(tutor_data_path).rglob('*.npy'))


def extract_data_from_filepath(relat_fp: PosixPath) \
        -> Tuple[int, float, int, int]:
    """
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
    """
    regex_str = 'records_chronics_lout_(.*)_dnthreshold_(.*)' + \
                '/records_chronic_(.*)_dayscomp_(.*).npy'
    line_disabled, dn_threshold, chronic_id, dayscomp = \
        re.search(regex_str, str(relat_fp)).groups()
    return int(line_disabled), float(dn_threshold), int(chronic_id), \
        int(dayscomp)


def extract_data_from_single_ts(ts_vect: np.array,
                                grid2op_vect_len: int,
                                vect2obs_func: Callable,
                                line_disabled: int,
                                env_info_dict: dict,
                                thermal_limits: Sequence[int]) -> dict:
    """
    Given the vector of a datapoint representing a single timestep, extract
    the interesting data from this vector and return it as a dictionary.

    Parameters
    ----------
    ts_vect : np.array
        The vector.
    grid2op_vect_len : int
        The length of the vector that represents a grid2op observation.
    vect2obs_func : Callable
        Function for transferring a vector representation of a grid2op
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
    """
    grid2op_obs_vect = ts_vect[-grid2op_vect_len:]
    obs = vect2obs_func(grid2op_obs_vect)
    obs_dict = obs.to_dict()

    data = {'action_index': int(ts_vect[0]),
            'timestep': int(ts_vect[4]),
            'gen_features': g2o_util.extract_gen_features(obs_dict),
            'load_features': g2o_util.extract_load_features(obs_dict),
            'or_features': g2o_util.extract_or_features(obs_dict, thermal_limits),
            'ex_features': g2o_util.extract_ex_features(obs_dict, thermal_limits),
            'topo_vect': obs_dict['topo_vect'].copy()
            }

    # Remove the disabled line from the data, if necessary
    if line_disabled != -1:
        data['or_features'] = np.delete(data['or_features'], line_disabled, axis=0)
        data['ex_features'] = np.delete(data['ex_features'], line_disabled, axis=0)
        data['topo_vect'] = np.delete(data['topo_vect'], [
            env_info_dict['dis_line_or_tv'],
            env_info_dict['dis_line_ex_tv']])

    # Assert the topo_vect has the same length as the features
    assert len(data['topo_vect']) == len(data['gen_features']) + \
           len(data['load_features']) + \
           len(data['or_features']) + \
           len(data['ex_features'])
    return data


def env_info_line_disabled(env: grid2op.Environment.Environment,
                           line_disabled: int) -> dict:
    """
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
    """
    sub_info = env.sub_info.copy()
    gen_pos_topo_vect = env.gen_pos_topo_vect.copy()
    load_pos_topo_vect = env.load_pos_topo_vect.copy()
    line_or_pos_topo_vect = env.line_or_pos_topo_vect.copy()
    line_ex_pos_topo_vect = env.line_ex_pos_topo_vect.copy()

    if line_disabled != -1:
        dis_line_or_tv = line_or_pos_topo_vect[line_disabled]
        dis_line_ex_tv = line_ex_pos_topo_vect[line_disabled]

        # Remove line at index from line_or/ex_pos_topo_vect
        line_or_pos_topo_vect = np.delete(line_or_pos_topo_vect, line_disabled)
        line_ex_pos_topo_vect = np.delete(line_ex_pos_topo_vect, line_disabled)

        # Lowering numbers in the sub_info array
        sub_info[env.line_or_to_subid[line_disabled]] -= 1
        sub_info[env.line_ex_to_subid[line_disabled]] -= 1

        # Lowering indices in the rest of the arrays indexing the topo_vect
        gen_pos_topo_vect = np.array([i - (i > dis_line_or_tv) -
                                      (i > dis_line_ex_tv) for i in gen_pos_topo_vect])
        load_pos_topo_vect = np.array([i - (i > dis_line_or_tv) -
                                       (i > dis_line_ex_tv) for i in load_pos_topo_vect])
        line_or_pos_topo_vect = np.array([i - (i > dis_line_or_tv) -
                                          (i > dis_line_ex_tv) for i in line_or_pos_topo_vect])
        line_ex_pos_topo_vect = np.array([i - (i > dis_line_or_tv) -
                                          (i > dis_line_ex_tv) for i in line_ex_pos_topo_vect])

    concat_ptvs = np.concatenate([gen_pos_topo_vect, load_pos_topo_vect,
                                  line_or_pos_topo_vect, line_ex_pos_topo_vect])
    # Check that the arrays indexing the topo vect are disjoint
    assert len(set(concat_ptvs)) == len(gen_pos_topo_vect) + len(load_pos_topo_vect) + \
           len(line_or_pos_topo_vect) + len(line_ex_pos_topo_vect)
    # Check that the sub_info max. index (plus one) equals the nr. of indices
    # equals the sum of objects
    assert max(concat_ptvs) + 1 == len(concat_ptvs) == sum(sub_info)

    info_dict = {'sub_info': sub_info,
                 'gen_pos_topo_vect': gen_pos_topo_vect,
                 'load_pos_topo_vect': load_pos_topo_vect,
                 'line_or_pos_topo_vect': line_or_pos_topo_vect,
                 'line_ex_pos_topo_vect': line_ex_pos_topo_vect}
    if line_disabled != -1:
        info_dict['dis_line_or_tv'] = dis_line_or_tv
        info_dict['dis_line_ex_tv'] = dis_line_ex_tv

    return info_dict


class ConMatrixCache:
    """
    Connectivity matrices are expensive to compute and store and many
    datapoints might share the same connectivity matrix. For this reason, we
    only compute/store each con. matrix once, and instead provide data points
    with a hash pointing to the correct con. matrix.
    """

    def __init__(self):
        self.con_matrices = {}

    def get_key_add_to_dict(self,
                            topo_vect: np.array,
                            line_disabled: int,
                            sub_info: np.array,
                            gen_pos_topo_vect: np.array,
                            load_pos_topo_vect: np.array,
                            line_or_pos_topo_vect: np.array,
                            line_ex_pos_topo_vect: np.array) -> int:
        """
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
        gen_pos_topo_vect : np.array
            Vector representing the indices of the generators in the topo_vect.
        load_pos_topo_vect : np.array
            Vector representing the indices of the loads in the topo_vect.
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

        """
        # Check that line_or_pos_topo_vect and line_ex_pos_topo_vect
        # have no overlap
        assert set(line_or_pos_topo_vect).isdisjoint(set(line_ex_pos_topo_vect))
        # And have the same size
        assert len(line_or_pos_topo_vect) == len(line_ex_pos_topo_vect)
        # Check that the number of objects according to sub_info and topo_vect
        # are equal
        assert sum(sub_info) == len(topo_vect)

        hash_topo_vect = hash((line_disabled, util.hash_nparray(topo_vect)))
        if hash_topo_vect not in self.con_matrices:
            con_matrices = g2o_util.connectivity_matrices(sub_info.astype(int),
                                                          topo_vect.astype(int),
                                                          line_or_pos_topo_vect.astype(int),
                                                          line_ex_pos_topo_vect.astype(int))
            hetero_con_matrices = g2o_util.connectivity_matrices_to_hetero_connectivity_matrices(
                list(gen_pos_topo_vect),
                list(load_pos_topo_vect),
                list(line_or_pos_topo_vect),
                list(line_ex_pos_topo_vect),
                {'same_busbar': con_matrices[0],
                 'other_busbar': con_matrices[1],
                 'line': con_matrices[2]})

            # Convert hetero_con_matrices keys from tuples to strings, because of json
            hetero_con_matrices = dict([(",".join(k), v) for k, v in hetero_con_matrices.items()])
            self.con_matrices[hash_topo_vect] = (topo_vect, con_matrices, hetero_con_matrices)

        return hash_topo_vect

    def save(self, fpath: str = ''):
        """
        Save the dictionary of connectivity matrices as a json file.

        Parameters
        ----------
        fpath : str, optional
            Where to store the json file. The default is the working directory.

        """
        with open(fpath, 'w') as outfile:
            json.dump(self.con_matrices, outfile, cls=NumpyEncoder)

    @classmethod
    def load(cls, fpath: str):
        """
        Factory class: initialize a ConMatrixCache based on a file.

        Parameters
        ----------
        fpath : str
            The filepath of the file.
        """
        cmc = cls()
        with open(fpath, 'r') as file:
            cmc.con_matrices = json.loads(file.read())

        # Change the hetero_con_matrices back from strings to tuples
        for cm in cmc.con_matrices.values():
            cm[2] = dict([(tuple(k.split(',')), v) for k, v in cm[2].items()])
        return cmc


class FeatureStatistics:
    """
    Used to track the statistics about features (N, mean, std), which are used
    in feature normalization.

    Since the dataset is too large to hold in memory completely, the feature
    statistics are computed iteratively.
    """

    def __init__(self):
        # Initialize statistics about features
        self.N_gen, self.N_load, self.N_line = 0, 0, 0
        self.S_gen, self.S_load, self.S_or, self.S_ex = None, None, None, None
        self.S2_gen, self.S2_load, self.S2_or, self.S2_ex = None, None, None, None

    def update_feature_statistics(self, data: dict):
        """
        Update the statistics (number, sum, sum of squares) of the feature
        values.

        Parameters
        ----------
        data : np.array
            Dictionary representing the datapoints, containing the features.
        """
        features = [data['gen_features'], data['load_features'],
                    data['or_features'], data['ex_features']]

        # Update number of objects
        self.N_gen, self.N_load, self.N_line = [n + f.shape[0] for f, n in
                                                zip(features[:-1], [self.N_gen, self.N_load, self.N_line])]

        if self.S_gen is None:
            # Initialize sum
            self.S_gen, self.S_load, self.S_or, self.S_ex = \
                [f.sum(axis=0) for f in features]
            # Initialize sum of squares
            self.S2_gen, self.S2_load, self.S2_or, self.S2_ex = \
                [(f ** 2).sum(axis=0) for f in features]
        else:
            # Increase the sum
            self.S_gen, self.S_load, self.S_or, self.S_ex = \
                [s + f.sum(axis=0) for f, s in zip(features,
                                                   [self.S_gen, self.S_load, self.S_or, self.S_ex])]
            # Increase the sum of squares
            self.S2_gen, self.S2_load, self.S2_or, self.S2_ex = \
                [s2 + (f ** 2).sum(axis=0) for f, s2 in zip(features,
                                                            [self.S2_gen, self.S2_load, self.S2_or, self.S2_ex])]

    def save(self, fpath: str):
        """
        Save the feature statistics in the form of the mean and standard
        deviation per object type to a specified location.

        Parameters
        ----------
        fpath : str
            The filepath to save to.
        """

        def std(num, sm, sum2):
            return np.sqrt(sum2 / num - (sm / num) ** 2)

        stats = {}
        for name, N, S, S2 in [('gen', self.N_gen, self.S_gen, self.S2_gen),
                               ('load', self.N_load, self.S_load, self.S2_load),
                               ('or', self.N_line, self.S_or, self.S2_or),
                               ('ex', self.N_line, self.S_ex, self.S2_ex)]:
            stats[name] = {'mean': S / N,
                           'std': std(N, S, S2)}
        with open(fpath, 'w') as outfile:
            json.dump(stats, outfile, cls=NumpyEncoder)


def process_raw_tutor_data():
    """
    Process the raw datapoints and store the processed datapoints.
    """

    # Specify paths
    config = get_config()
    raw_data_path = config['paths']['data']['raw']
    processed_data_path = config['paths']['data']['processed']
    split_path = config['paths']['data_split']

    # Create subdirectories
    create_subdirectories()

    # Load train, val, test sets
    train_scenarios = np.load(split_path + 'train_scenarios.npy')
    val_scenarios = np.load(split_path + 'val_scenarios.npy')
    test_scenarios = np.load(split_path + 'test_scenarios.npy')

    # con_matrix_path = processed_data_path + '/auxiliary_data_objects/con_matrix_cache.json'
    # fstats_path = processed_data_path + '/auxiliary_data_objects/feature_statistics.json'

    # Initialize environment and environment variables
    env = g2o_util.init_env(grid2op.Rules.AlwaysLegal)
    grid2op_vect_size = len(env.get_obs().to_vect())
    thermal_limits = config['rte_case14_realistic']['thermal_limits']

    # Create an object for caching connectivity matrices
    cmc = ConMatrixCache()
    # Create a dictionary used for finding actions corresponding to action ids
    action_iders = {}
    # Create object for tracking the feature statistics
    fstats = FeatureStatistics()

    filepaths = get_filepaths(raw_data_path)
    random.shuffle(filepaths)
    for filepath in tqdm(filepaths):
        line_disabled, _, chronic_id, days_completed = \
            extract_data_from_filepath(filepath.relative_to(raw_data_path))

        # Load a single file with raw datapoints
        raw_datapoints = np.load(filepath)

        # Env information specifically for a line removed
        env_info_dict = env_info_line_disabled(env, line_disabled)

        # If it doesn't already exit, create action_identificator for this
        # particular line disabled
        # Action identificator give the action corresponding to an action index
        if line_disabled not in action_iders:
            action_iders[line_disabled] = action_identificator(env, line_disabled)

        # Loop over the datapoints
        for raw_dp in raw_datapoints:
            # Extract information dictionary from the datapoint
            dp = extract_data_from_single_ts(raw_dp,
                                             grid2op_vect_size,
                                             env.observation_space.from_vect,
                                             line_disabled,
                                             env_info_dict,
                                             thermal_limits)

            # Add the data from the filepath and environment to the data dictionary
            dp.update({'line_disabled': line_disabled,
                       'chronic_id': chronic_id,
                       'dayscomp': days_completed})
            dp.update({'sub_info': env_info_dict['sub_info'],
                       'gen_pos_topo_vect': env_info_dict['gen_pos_topo_vect'],
                       'load_pos_topo_vect': env_info_dict['load_pos_topo_vect'],
                       'line_or_pos_topo_vect': env_info_dict['line_or_pos_topo_vect'],
                       'line_ex_pos_topo_vect': env_info_dict['line_ex_pos_topo_vect'],
                       })

            # Update the feature statistics if the file is in the train partition
            if chronic_id in train_scenarios:
                fstats.update_feature_statistics(dp)

            # Find the set action topology vector and add it to the datapoint
            if dp['action_index'] != -1:
                action_ider = action_iders[line_disabled]
                dp['set_topo_vect'] = action_ider.get_set_topo_vect(dp['action_index'])
                # Remove disables lines from topo vect objects
                if line_disabled != -1:
                    dp['set_topo_vect'] = np.delete(dp['set_topo_vect'], [
                        env_info_dict['dis_line_or_tv'],
                        env_info_dict['dis_line_ex_tv']])
            else:
                dp['set_topo_vect'] = np.zeros_like(dp['topo_vect'])

            dp['change_topo_vect'] = np.array([0 if s == 0 else abs(t - s) for t, s in
                                               zip(dp['topo_vect'], dp['set_topo_vect'])])
            dp['res_topo_vect'] = np.array([t if s == 0 else s for t, s in
                                            zip(dp['topo_vect'], dp['set_topo_vect'])])

            # Skip datapoint if any other line is disabled
            if -1 in dp['topo_vect']:
                continue

            assert len(dp['set_topo_vect']) == len(dp['topo_vect']) == len(dp['change_topo_vect']) \
                   == len(dp['res_topo_vect']), "Not equal lengths"
            assert len(dp['topo_vect']) == (56 if line_disabled == -1 else 54), \
                "Incorrect length"
            assert all([(o in [0, 1, 2]) for o in dp['set_topo_vect']]), \
                "Incorrect element in set_topo_vect"
            assert all([(o in [1, 2]) for o in dp['topo_vect']]), \
                "Incorrect element in topo_vect"
            assert all([(o in [0, 1]) for o in dp['change_topo_vect']]), \
                "Incorrect element in change_topo_vect"
            assert all([(o in [1, 2]) for o in dp['res_topo_vect']]), \
                "Incorrect element in res_topo_vect"

            # Add the index of the connectivity matrix to the data object
            cm_index = cmc.get_key_add_to_dict(dp['topo_vect'],
                                               line_disabled,
                                               env_info_dict['sub_info'],
                                               env_info_dict['gen_pos_topo_vect'],
                                               env_info_dict['load_pos_topo_vect'],
                                               env_info_dict['line_or_pos_topo_vect'],
                                               env_info_dict['line_ex_pos_topo_vect'])
            dp['cm_index'] = cm_index
            assert dp['cm_index'] in cmc.con_matrices

            # Add datapoint to random datafile
            if chronic_id in train_scenarios:
                datafile_number = random.randint(0, config['dataset']['n_train_datafiles'] - 1)
                filepath = processed_data_path + f'train/data_{datafile_number}.json'
                save_datapoint_to_file(dp, filepath)
            elif chronic_id in val_scenarios:
                datafile_number = random.randint(0, config['dataset']['n_val_datafiles'] - 1)
                filepath = processed_data_path + f'val/data_{datafile_number}.json'
                save_datapoint_to_file(dp, filepath)
            elif chronic_id in test_scenarios:
                datafile_number = random.randint(0, config['dataset']['n_test_datafiles'] - 1)
                filepath = processed_data_path + f'test/data_{datafile_number}.json'
                save_datapoint_to_file(dp, filepath)
            else:
                # Discard datapoint if not in any of the three partitions
                continue

    # Save auxiliary data objects
    cmc.save(processed_data_path + 'auxiliary_data_objects/con_matrix_cache.json')
    fstats.save(processed_data_path + 'auxiliary_data_objects/feature_stats.json')


def save_datapoint_to_file(datapoint: dict, filepath: str):
    """
    Given a dictionary representing a data point, append it to a json datafile identified by the filepath.
    Creates the file if it does not exist.

    Parameters
    ----------
    datapoint : dict
       The datapoint.
    filepath : str
       The filepath.
    """
    if os.path.isfile(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
    else:
        data = []

    data.append(datapoint)

    with open(filepath, "w") as file:
        json.dump(data, file, cls=NumpyEncoder)


def create_subdirectories():
    """
    Create the train, val, test, and auxiliary_data_objects subdirectories.
    Overwrites them if they already exist and contain only .json files.

    Raises
    ------
    AssertationError
        Whenever there are files in the existing train/val/test folders which are not .json files.
    """
    config = get_config()
    processed_path = config['paths']['data']['processed']

    # Remove directories including existing processed datapoints
    for name in ['train', 'val', 'test', 'auxiliary_data_objects']:
        if os.path.exists(processed_path + name):
            assert all([file.endswith('.json') for file in
                        os.listdir(processed_path + name)]), \
                f'All files in the {name} folder to be overwritten must be .json files.'

            # Remove directory recursively
            shutil.rmtree(processed_path + name)

        # Create new directory
        os.mkdir(processed_path + name)


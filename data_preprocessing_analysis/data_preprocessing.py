#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:30:55 2021

@author: matthijs
"""
import random

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
        self.S_gen, self.S_load, self.S_line = None, None, None
        self.S2_gen, self.S2_load, self.S2_line = None, None, None

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
            # Initialize generator statistics
            self.S_gen = features[0].sum(axis=0)
            self.S2_gen = (features[0] ** 2).sum(axis=0)

            # Initialize load statistics
            self.S_load = features[1].sum(axis=0)
            self.S2_load = (features[1] ** 2).sum(axis=0)

            # Initialize line statistics
            self.S_line = features[2].sum(axis=0) + features[3].sum(axis=0)
            self.S2_line = (features[2] ** 2).sum(axis=0) + (features[3] ** 2).sum(axis=0)
        else:
            # Update generator statistics
            self.S_gen += features[0].sum(axis=0)
            self.S2_gen += (features[0] ** 2).sum(axis=0)

            # Update load statistics
            self.S_load += features[1].sum(axis=0)
            self.S2_load += (features[1] ** 2).sum(axis=0)

            # Initialize line statistics
            self.S_line += features[2].sum(axis=0) + features[3].sum(axis=0)
            self.S2_line += (features[2] ** 2).sum(axis=0) + (features[3] ** 2).sum(axis=0)

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
                               ('line', 2 * self.N_line, self.S_line, self.S2_line)]:
            stats[name] = {'mean': S / N,
                           'std': std(N, S, S2)}
        with open(fpath, 'w') as outfile:
            json.dump(stats, outfile, cls=NumpyEncoder)


def process_raw_data():
    """
    Process the raw datapoints and store the resulting processed datapoints.
    """
    # Specify paths
    config = get_config()
    raw_data_path = config['paths']['data']['raw']
    processed_data_path = config['paths']['data']['processed']
    split_path = config['paths']['data_split']
    buffer_size = config['data_processing']['buffer_size']
    output_file_size = config['data_processing']['output_file_size']

    # Create subdirectories where the data will be stored
    create_subdirectories()

    # Load train, val, test sets
    train_scenarios = np.load(split_path + 'train_scenarios.npy')
    val_scenarios = np.load(split_path + 'val_scenarios.npy')
    test_scenarios = np.load(split_path + 'test_scenarios.npy')

    # Initialize environment and environment variables
    env = g2o_util.init_env()
    grid2op_vect_size = len(env.get_obs().to_vect())

    # Create an object for caching connectivity matrices
    con_matrix_cache = ConMatrixCache()
    # Create a dictionary of action_identificators for retrieving Grid2Op actions from action indices
    action_iders = {}
    # Create object for tracking the feature statistics
    feature_stats = FeatureStatistics()

    # Generate a dataset for each data partition
    for scenarios, folder_name in [(train_scenarios, 'train'),
                                   (val_scenarios, 'val'),
                                   (test_scenarios, 'test')]:
        filepaths = get_filepaths(raw_data_path)
        random.shuffle(filepaths)

        buffer = []
        filepath_i = 0

        # Process each file
        for filepath in tqdm(filepaths):
            # Extract information encoded in the filepath
            line_disabled, _, chronic_id, days_completed = \
                extract_data_from_filepath(filepath.relative_to(raw_data_path))

            # Skip scenario if it is not in the current data partition
            if chronic_id not in scenarios:
                continue

            # Load a single file containing raw datapoints
            raw_datapoints = np.load(filepath)

            # If not yet existing, create an action_identificator for this particular line disabled
            if line_disabled not in action_iders:
                action_iders[line_disabled] = action_identificator(env, line_disabled)

            # Obtain the reduced env variables for this line disabled
            env_var_dict = reduced_env_variables([line_disabled] if line_disabled != -1 else [])

            # Loop over the raw datapoints in file
            for raw_dp in raw_datapoints:
                # Extract data from raw datapoint
                dp = extract_raw_dp_data(raw_dp, grid2op_vect_size, env.observation_space.from_vect)

                # Add the filepath and environment data to the datapoint
                dp.update({'line_disabled': line_disabled, 'chronic_id': chronic_id, 'dayscomp': days_completed})
                dp.update({'sub_info': env_var_dict['sub_info'],
                           'gen_pos_topo_vect': env_var_dict['gen_pos_topo_vect'],
                           'load_pos_topo_vect': env_var_dict['load_pos_topo_vect'],
                           'line_or_pos_topo_vect': env_var_dict['line_or_pos_topo_vect'],
                           'line_ex_pos_topo_vect': env_var_dict['line_ex_pos_topo_vect']
                           })

                # Remove disabled lines in datapoint by reducing variables
                reduce_variables_datapoint(dp)

                # Update the feature statistics if the file is in the train partition
                if (folder_name == 'train') and (chronic_id in train_scenarios):
                    feature_stats.update_feature_statistics(dp)

                # Retrieve the set-action, potentially reduce it
                if dp['action_index'] != -1:
                    action_ider = action_iders[line_disabled]
                    dp['set_topo_vect'] = action_ider.get_set_topo_vect(dp['action_index'])

                    # Reduce the set-action vector if any lines are disabled
                    if line_disabled != -1:
                        dis_line_or_tv = config['rte_case14_realistic']['line_or_pos_topo_vect'][dp['line_disabled']]
                        dis_line_ex_tv = config['rte_case14_realistic']['line_ex_pos_topo_vect'][dp['line_disabled']]

                        dp['set_topo_vect'] = np.delete(dp['set_topo_vect'], (dis_line_or_tv, dis_line_ex_tv))
                else:
                    dp['set_topo_vect'] = np.zeros_like(dp['topo_vect'])

                # Calculate the change-action and the resulting topology
                dp['change_topo_vect'] = np.array([0 if s == 0 else abs(t - s) for t, s in
                                                   zip(dp['topo_vect'], dp['set_topo_vect'])])
                dp['res_topo_vect'] = np.array([t if s == 0 else s for t, s in
                                                zip(dp['topo_vect'], dp['set_topo_vect'])])

                # Skip datapoint if any other line is disabled
                if -1 in dp['topo_vect']:
                    continue

                # Postconditions on the created variables
                assert len(dp['set_topo_vect']) == len(dp['topo_vect']) == len(dp['change_topo_vect']) \
                       == len(dp['res_topo_vect']), "Not equal lengths"
                assert all([(o in [0, 1, 2]) for o in dp['set_topo_vect']]), "Incorrect element in set_topo_vect"
                assert all([(o in [1, 2]) for o in dp['topo_vect']]), "Incorrect element in topo_vect"
                assert all([(o in [0, 1]) for o in dp['change_topo_vect']]), "Incorrect element in change_topo_vect"
                assert all([(o in [1, 2]) for o in dp['res_topo_vect']]), "Incorrect element in res_topo_vect"

                # Add the index of the connectivity matrix to the datapoint
                con_matrix_index = con_matrix_cache.get_key_add_to_dict(dp['topo_vect'],
                                                                        line_disabled,
                                                                        env_var_dict['sub_info'],
                                                                        env_var_dict['gen_pos_topo_vect'],
                                                                        env_var_dict['load_pos_topo_vect'],
                                                                        env_var_dict['line_or_pos_topo_vect'],
                                                                        env_var_dict['line_ex_pos_topo_vect'])
                dp['cm_index'] = con_matrix_index
                assert dp['cm_index'] in con_matrix_cache.con_matrices

                # Add the datapoint to the buffer
                buffer.append(dp)

                # If the buffer is full, save datapoints
                if len(buffer) >= buffer_size:
                    save_dps = [buffer.pop(random.randrange(len(buffer))) for _ in range(output_file_size)]
                    filepath = processed_data_path + f'{folder_name}/data_{filepath_i}.json'
                    save_datapoints(save_dps, filepath)
                    filepath_i += 1

            # Save datapoints until the buffer is empty
            while len(buffer) > 0:
                save_dps = [buffer.pop(random.randrange(len(buffer))) for _ in
                            range(min(output_file_size, len(buffer)))]
                filepath = processed_data_path + f'{folder_name}/data_{filepath_i}.json'
                save_datapoints(save_dps, filepath)
                filepath_i += 1

    # Save auxiliary data objects
    con_matrix_cache.save(processed_data_path + 'auxiliary_data_objects/con_matrix_cache.json')
    feature_stats.save(processed_data_path + 'auxiliary_data_objects/feature_stats.json')


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
    return list(Path(tutor_data_path).rglob('*/*.npy'))


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


def extract_raw_dp_data(ts_vect: np.array, grid2op_vect_len: int, vect2obs_func: Callable) -> dict:
    """
    Given a vector representing a single datapoint, extract the relevant data from this vector and return it in
    dictionary form.

    Parameters
    ----------
    ts_vect : np.array
        The vector.
    grid2op_vect_len : int
        The length of the vector that represents a grid2op observation.
    vect2obs_func : Callable
        Function for transferring a vector representation of a grid2op
        observation to the corresponding grid2op observation object.

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
            'gen_features': extract_gen_features(obs_dict),
            'load_features': extract_load_features(obs_dict),
            'or_features': extract_or_features(obs_dict),
            'ex_features': extract_ex_features(obs_dict),
            'topo_vect': obs_dict['topo_vect'].copy()
            }

    # Assert the topo_vect has the same length as the features
    assert len(data['topo_vect']) == len(data['gen_features']) + len(data['load_features']) + \
           len(data['or_features']) + len(data['ex_features'])

    return data


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


def extract_or_features(obs_dict: dict) \
        -> np.array:
    """
    Given the grid2op observation in dictionary form,
    return the load features.

    Parameters
    ----------
    obs_dict : dict
        Dictionary representation of the grid2op observation. Can be obtained
        with obs.to_dict().

    Returns
    -------
    X : np.array
        Array representation of the features;  rows correspond to the different objects. Columns represent the
        'p', 'q', 'v', 'a', 'line_rho', 'thermal limit' features.
    """
    X = np.array(list(obs_dict['lines_or'].values())).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.concatenate((X,
                            np.reshape(np.array(obs_dict['rho']), (-1, 1)),
                            np.reshape(np.array(obs_dict['lines_or']['p']) / np.array(obs_dict['rho']), (-1, 1))),
                           axis=1)
    return X


def extract_ex_features(obs_dict: dict) \
        -> np.array:
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
        Array representation of the features; rows correspond to the different objects. Columns represent the
        'p', 'q', 'v', 'a', 'line_rho', 'thermal limit' features.
    """
    X = np.array(list(obs_dict['lines_ex'].values())).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.concatenate((X,
                            np.reshape(np.array(obs_dict['rho']), (-1, 1)),
                            np.reshape(np.array(obs_dict['lines_ex']['p']) / np.array(obs_dict['rho']), (-1, 1))),
                           axis=1)
    return X


def reduce_variables_datapoint(datapoint: dict):
    """

    :param datapoint:
    :return:
    """
    config = get_config()
    line_disabled = datapoint['line_disabled']

    # Reduce variables if a line is disabled
    if line_disabled != -1:
        dis_line_or_tv = config['rte_case14_realistic']['line_or_pos_topo_vect'][datapoint['line_disabled']]
        dis_line_ex_tv = config['rte_case14_realistic']['line_ex_pos_topo_vect'][datapoint['line_disabled']]

        # Check and reduce the topo_vect variable
        assert datapoint['topo_vect'][dis_line_or_tv] == -1, 'Disabled origin should be -1.'
        assert datapoint['topo_vect'][dis_line_ex_tv] == -1, 'Disabled extremity should be -1.'
        datapoint['topo_vect'] = np.delete(datapoint['topo_vect'], (dis_line_or_tv, dis_line_ex_tv))

        # Check and reduce the line endpoint features
        assert np.isclose(datapoint['or_features'][line_disabled, :-1], 0).all(), ("All features except the last "
                                                                                   "of the disabled origin must be "
                                                                                   "zero.")
        assert np.isclose(datapoint['ex_features'][line_disabled, :-1], 0).all(), ("All features except the last "
                                                                                   "of the disabled extremity must be "
                                                                                   "zero.")
        datapoint['or_features'] = np.delete(datapoint['or_features'], line_disabled, axis=0)
        datapoint['ex_features'] = np.delete(datapoint['ex_features'], line_disabled, axis=0)
        assert datapoint['or_features'].shape == datapoint['ex_features'].shape, ("Origin and extremities' features "
                                                                                  "should have the same shape.")


def reduced_env_variables(lines_disabled: list[int]) -> dict:
    """
    Returns Grid2Op variables that reflect completely removing the disables lines (given by 'lines_disabled') from the
    environment. This consists of altering the 'sub_info' and '[...]_pos_topo_vect' variables. If 'lines_disabled' is
    empty, these variables are returned unaltered.

    Parameters
    ----------
    lines_disabled : list[int]
        The indices of the disabled lines.

    Returns
    -------
    dict
        The dictionary with the altered variables:
            'sub_info': the number of elements per substation.
            'pos_topo_vect': aggregation of the four succeeding entries.
            'gen_pos_topo_vect': indices in the topo vect for each generator object.
            'load_pos_topo_vect': indices in the topo vect for each load object.
            'line_or_pos_topo_vect': indices in the topo vect for each origin object.
            'line_ex_pos_topo_vect': indices in the topo vect for each extremity object.
    """
    # Copy variables
    config = get_config()
    sub_info = np.array(config['rte_case14_realistic']['sub_info'])
    gen_pos_topo_vect = np.array(config['rte_case14_realistic']['gen_pos_topo_vect'])
    load_pos_topo_vect = np.array(config['rte_case14_realistic']['load_pos_topo_vect'])
    line_or_pos_topo_vect = np.array(config['rte_case14_realistic']['line_or_pos_topo_vect'])
    line_ex_pos_topo_vect = np.array(config['rte_case14_realistic']['line_ex_pos_topo_vect'])
    line_or_to_subid = np.array(config['rte_case14_realistic']['line_or_to_subid'])
    line_ex_to_subid = np.array(config['rte_case14_realistic']['line_ex_to_subid'])

    # Asserting preconditions
    assert len(lines_disabled) == len(set(lines_disabled)), "Duplicate line in variable lines_disabled."
    assert set(lines_disabled).issubset(set(range(len(line_or_pos_topo_vect)))), "Incorrect disabled line index."

    # Reduce variables if there are disconnected lines
    if len(lines_disabled) > 0:

        # Find the indices in the topology vector of the disabled line endpoints
        disabled_ors_idxs = [line_or_pos_topo_vect[line] for line in lines_disabled]
        disabled_exs_idxs = [line_ex_pos_topo_vect[line] for line in lines_disabled]

        # Remove line at index from line_or/ex_pos_topo_vect
        line_or_pos_topo_vect = np.delete(line_or_pos_topo_vect, lines_disabled)
        line_ex_pos_topo_vect = np.delete(line_ex_pos_topo_vect, lines_disabled)

        # Decrementing elements in the sub_info array
        for line in lines_disabled:
            sub_info[line_or_to_subid[line]] -= 1
            sub_info[line_ex_to_subid[line]] -= 1

        # Lowering indices in the pos_topo_vect arrays to compensate for the deleted variables
        combined_disabled_idxs = np.concatenate((disabled_ors_idxs, disabled_exs_idxs))
        gen_pos_topo_vect = np.array([i - sum(i > combined_disabled_idxs) for i in gen_pos_topo_vect]).flatten()
        load_pos_topo_vect = np.array([i - sum(i > combined_disabled_idxs) for i in load_pos_topo_vect]).flatten()
        line_or_pos_topo_vect = np.array([i - sum(i > combined_disabled_idxs) for i in line_or_pos_topo_vect]).flatten()
        line_ex_pos_topo_vect = np.array([i - sum(i > combined_disabled_idxs) for i in line_ex_pos_topo_vect]).flatten()

    # Concatenating the pos_topo_vect variables
    pos_topo_vect = np.concatenate([gen_pos_topo_vect, load_pos_topo_vect,
                                    line_or_pos_topo_vect, line_ex_pos_topo_vect])

    # Asserting postconditions
    assert sum(sub_info) == len(pos_topo_vect), "The sub_info and pos_topo_vect imply a different number of objects."
    assert np.all(np.sort(pos_topo_vect) == np.arange(0, len(pos_topo_vect))), ("The elements of pos_topo_vect must "
                                                                                "map onto the indices of it's range.")

    # Returning the dictionary of variables
    var_dict = {'sub_info': sub_info,
                'pos_topo_vect': pos_topo_vect,
                'gen_pos_topo_vect': gen_pos_topo_vect,
                'load_pos_topo_vect': load_pos_topo_vect,
                'line_or_pos_topo_vect': line_or_pos_topo_vect,
                'line_ex_pos_topo_vect': line_ex_pos_topo_vect}

    return var_dict


def save_datapoints(datapoints: Sequence[dict], filepath: str):
    """
    Given a list of datapoints, save it as a json file to the specified filepath.

    Parameters
    ----------
    datapoints : Sequence[dict]
       The list of datapoints.
    filepath : str
       The filepath.
    """
    with open(filepath, "w") as file:
        json.dump(datapoints, file, cls=NumpyEncoder)

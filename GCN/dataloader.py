#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 09:49:36 2022

@author: matthijs
"""
import torch
import os
import json
import random
import data_preprocessing_analysis.imitation_data_preprocessing as idp
from typing import List
import numpy as np


class TutorDataLoader:
    """
    Object for loading the tutor dataset.
    """

    def __init__(self,
                 root: str,
                 matrix_cache_path: str,
                 feature_statistics_path: str,
                 action_counter_path: str,
                 device: torch.device,
                 network_type: str,
                 train: bool,
                 action_frequency_threshold: int = 0):
        """
        Parameters
        ----------
        root : str
            The directory where the data files are located.
        matrix_cache_path : str
            The path of the matrix cache file.
        feature_statistics_path : str
            The path of the feature statistics file.
        action_counter_path: str
            The path of the action counter json file.
        device : torch.device
            What device to load the data on.
        network_type : str
            The type of network. Based on this, data needs to be delivered
            in a different format. Should be 'homogeneous' or 'heterogeneous'.
        train : bool
            Whether the loaded data is used for training or validation.
            More information is included in validation.
        action_frequency_threshold : int
            Mimimum frequency of an action in the dataset in order to be
            used during training. Can be used to filter out infrequent actions.
            Default is zero.
        """

        self._file_names = os.listdir(root)
        self._file_paths = [os.path.join(root, fn) for fn in self._file_names]
        self._matrix_cache = idp.ConMatrixCache.load(matrix_cache_path)
        with open(feature_statistics_path, 'r') as file:
            self._feature_statistics = json.loads(file.read())
        with open(action_counter_path, 'r') as file:
            self._action_counter = json.loads(file.read())
        self.device = device

        assert network_type in ['homogeneous', 'heterogeneous'], \
            'Invalid network type'
        self.network_type = network_type
        self.train = train
        self.action_frequency_threshold = action_frequency_threshold

    def get_file_datapoints(self, idx: int) -> List[dict]:
        """
        Load the datapoints in a particular file. The file is indexed by an
        int, representing the index of the file in the list of file paths.

        Parameters
        ----------
        idx : int
            The index of the file in the list of file paths.

        Returns
        -------
        processed_datapoints : List[dict]
            The list of datapoints. Each datapoint is a dictionary.
        """
        device = self.device

        # 'raw' is not fully true, as these datapoints should already have been
        # preprocessed

        with open(self._file_paths[idx], 'r') as file:
            raw_datapoints = json.loads(file.read())

        processed_datapoints = []
        for raw_dp in raw_datapoints:
            dp = dict()

            act_freq = self._action_counter[str(raw_dp['act_hash'])]
            if act_freq < self.action_frequency_threshold:
                continue

            # Load the label
            dp['change_topo_vect'] = torch.tensor(raw_dp['change_topo_vect'],
                                                  device=device,
                                                  dtype=torch.float)

            # Create the object position topology vector, which relates the
            # objects ordered by type to their position in the topology vector
            dp['object_ptv'] = np.argsort(np.concatenate(
                [raw_dp['gen_pos_topo_vect'],
                 raw_dp['load_pos_topo_vect'],
                 raw_dp['line_or_pos_topo_vect'],
                 raw_dp['line_ex_pos_topo_vect']]))

            # Load the sub info array, which contains info about to which
            # substation each object belongs
            dp['sub_info'] = raw_dp['sub_info']

            # Load, normalize features, turn them into tensors
            fstats = self._feature_statistics
            norm_gen_features = (np.array(raw_dp['gen_features'])
                                 - fstats['gen']['mean']) / fstats['gen']['std']
            dp['gen_features'] = torch.tensor(norm_gen_features,
                                              device=device,
                                              dtype=torch.float)
            norm_load_features = (np.array(raw_dp['load_features'])
                                  - fstats['load']['mean']) / fstats['load']['std']
            dp['load_features'] = torch.tensor(norm_load_features,
                                               device=device,
                                               dtype=torch.float)
            norm_or_features = (np.array(raw_dp['or_features'])
                                - fstats['or']['mean']) / fstats['or']['std']
            dp['or_features'] = torch.tensor(norm_or_features,
                                             device=device,
                                             dtype=torch.float)
            norm_ex_features = (np.array(raw_dp['ex_features'])
                                - fstats['ex']['mean']) / fstats['ex']['std']
            dp['ex_features'] = torch.tensor(norm_ex_features,
                                             device=device,
                                             dtype=torch.float)

            # Load the connectivity matrix, combine the edges for the specified
            # network type
            same_busbar_e, other_busbar_e, line_e = \
                self._matrix_cache.con_matrices[str(raw_dp['cm_index'])][1]
            if self.network_type == 'homogeneous':
                dp['edges'] = torch.tensor(np.append(same_busbar_e, line_e, axis=1),
                                           device=device,
                                           dtype=torch.long)
            elif self.network_type == 'heterogeneous':
                dp['edges'] = {('object', 'line', 'object'):
                                   torch.tensor(line_e,
                                                device=device,
                                                dtype=torch.long),
                               ('object', 'same_busbar', 'object'):
                                   torch.tensor(same_busbar_e,
                                                device=device,
                                                dtype=torch.long),
                               ('object', 'other_busbar', 'object'):
                                   torch.tensor(other_busbar_e,
                                                device=device,
                                                dtype=torch.long)}

            # If the data is not for training, add information used in
            # validation analysis
            if not self.train:
                dp['line_disabled'] = raw_dp['line_disabled']
                dp['topo_vect'] = torch.tensor(raw_dp['topo_vect'],
                                               device=device,
                                               dtype=torch.long)

            processed_datapoints.append(dp)

        return processed_datapoints

    def __iter__(self, shuffle: bool = True) -> dict:
        """
        Iterate over the datapoints.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data files. Does NOT mean that the dps
            in the files are also shuffled. The default is True.

        Yields
        ------
        dp : dict
            The datapoint.

        """
        file_idxs = list(range(len(self._file_paths)))
        if shuffle:
            random.shuffle(file_idxs)

        for idx in file_idxs:
            datapoints = self.get_file_datapoints(idx)
            for i, dp in enumerate(datapoints):
                yield dp

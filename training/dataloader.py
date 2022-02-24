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
from typing import List, Optional, Type
import numpy as np
from training.models import GCN, FCNN
from abc import ABC, abstractmethod
from auxiliary.config import config


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
                 model_type: Type,
                 network_type: Optional[GCN.NetworkType],
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
        network_type : NetworkType
            The type of network. Based on this, data needs to be delivered
            in a different format.
        train : bool
            Whether the loaded data is used for training or validation.
            More information is included in validation.
        action_frequency_threshold : int
            Minimum frequency of an action in the dataset in order to be
            used during training. Can be used to filter out infrequent actions.
            Default is zero.
        """

        self._file_names = os.listdir(root)
        self._file_paths = [os.path.join(root, fn) for fn in self._file_names]
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        if model_type == GCN:
            assert isinstance(network_type, GCN.NetworkType), 'Invalid network type'
            matrix_cache = idp.ConMatrixCache.load(matrix_cache_path)
            self.process_dp_strategy = ProcessDataPointGCN(device,
                                                           train,
                                                           feature_statistics,
                                                           network_type,
                                                           matrix_cache)
        elif model_type == FCNN:
            self.process_dp_strategy = ProcessDataPointFCNN(device,
                                                            train,
                                                            feature_statistics)


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
        # 'raw' is not fully true, as these datapoints should already have been
        # preprocessed
        with open(self._file_paths[idx], 'r') as file:
            raw_datapoints = json.loads(file.read())

        processed_datapoints = []
        for raw_dp in raw_datapoints:

            # skip datapoints that occur too infrequently in the dataset
            # TODO: decide whether to keep this frequency check in
            # act_freq = self._action_counter[str(raw_dp['act_hash'])]
            # if act_freq < self.action_frequency_threshold:
            #    continue

            # process datapoint
            dp = self.process_dp_strategy.process_datapoint(raw_dp)

            # add processed datapoint to file list
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


class ProcessDataPointStrategy(ABC):
    """
    Abstract base class for strategies of processing a single datapoint.
    Exists because different models require different information from a datapoint,
    and hence require different processing.
    """

    def __init__(self,
                 device: torch.device,
                 train: bool,
                 feature_statistics: dict):
        """
        Parameters
        ----------
        device : torch.device
            The device to set torch tensors to.
        train : bool
            Whether to process the datapoint for training or not. More information is included for validation/testing.
        feature_statistics : dict
            Dictionary with information (mean, std) used to normalize features.
        """
        self.device = device
        self.train = train
        self.feature_statistics = feature_statistics
        self.class_weight_assigner = ClassWeightAssigner(config['paths']['action_counter'],
                                                         config['training']['hyperparams']['class_weights']
                                                               ['max_adapt_weight'],
                                                         config['training']['hyperparams']['class_weights']
                                                               ['min_adapt_weight'],
                                                         config['training']['hyperparams']['class_weights']
                                                               ['min_weight_zero'])

    @abstractmethod
    def process_datapoint(self, raw_dp: int):
        """
        Process a single datapoint, from raw_dp to dp.

        Parameters
        ----------
        raw_dp : dict
            The 'raw' datapoint (not fully true; these raw datapoints should be preprocessed already) from
            which information is extracted.

        Returns
        -------
        dp : dict
            The resulting datapoint.
        """
        pass

    def add_processed_label(self, raw_dp: dict, dp: dict):
        """
        Extract the label from raw_dp, process it, and store it in dp. Includes the class weight too.

        Parameters
        ----------
        raw_dp : dict
            The raw datapoint.
        dp : dict
            The processed datapoint to add information to.

        Returns
        -------
            dp : dict
                The processed datapoint with the processed label added.
        """
        dp['change_topo_vect'] = torch.tensor(raw_dp['change_topo_vect'],
                                              device=self.device,
                                              dtype=torch.float)
        dp['class_weight'] = self.class_weight_assigner.assign(raw_dp['act_hash'])
        return dp

    def add_val_info(self, raw_dp: dict, dp: dict):
        """
        Extract information used during evaluation from raw_dp, process it, and store it in dp.

        Parameters
        ----------
        raw_dp : dict
            The raw datapoint.
        dp : dict
            The processed datapoint to add information to.

        Returns
        -------
            dp : dict
                The processed datapoint with the processed evaluation information added.
        """
        dp['line_disabled'] = raw_dp['line_disabled']
        dp['topo_vect'] = torch.tensor(raw_dp['topo_vect'],
                                       device=self.device,
                                       dtype=torch.long)
        return dp


class ProcessDataPointGCN(ProcessDataPointStrategy):
    """
    Process a datapoint to obtain the information used by, and in the format used by, a GCN.
    """

    def __init__(self,
                 device: torch.device,
                 train: bool,
                 feature_statistics: dict,
                 network_type: GCN.NetworkType,
                 matrix_cache: idp.ConMatrixCache):
        """
        Parameters
        ----------
        device : torch.device
            The device to set torch tensors to.
        train : bool
            Whether to process the datapoint for training or not. More information is included for validation/testing.
        feature_statistics : dict
            Dictionary with information (mean, std) used to normalize features.
        network_type : NetworkType
            The type of the GCN network.
        """
        super().__init__(device, train, feature_statistics)
        self.network_type = network_type
        self.matrix_cache = matrix_cache

    def process_datapoint(self, raw_dp: dict):
        """
        Process a single datapoint, from raw_dp to dp, with the information and formatting for a GCN model.

        Parameters
        ----------
        raw_dp : dict
            The 'raw' datapoint (not fully true; these raw datapoints should be preprocessed already) from
            which information is extracted.

        Returns
        -------
        dp : dict
            The resulting datapoint.
        """
        dp = {}

        # Add the label
        dp = self.add_processed_label(raw_dp, dp)

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
        fstats = self.feature_statistics
        norm_gen_features = (np.array(raw_dp['gen_features'])
                             - fstats['gen']['mean']) / fstats['gen']['std']
        dp['gen_features'] = torch.tensor(norm_gen_features,
                                          device=self.device,
                                          dtype=torch.float)
        norm_load_features = (np.array(raw_dp['load_features'])
                              - fstats['load']['mean']) / fstats['load']['std']
        dp['load_features'] = torch.tensor(norm_load_features,
                                           device=self.device,
                                           dtype=torch.float)
        norm_or_features = (np.array(raw_dp['or_features'])
                            - fstats['or']['mean']) / fstats['or']['std']
        dp['or_features'] = torch.tensor(norm_or_features,
                                         device=self.device,
                                         dtype=torch.float)
        norm_ex_features = (np.array(raw_dp['ex_features'])
                            - fstats['ex']['mean']) / fstats['ex']['std']
        dp['ex_features'] = torch.tensor(norm_ex_features,
                                         device=self.device,
                                         dtype=torch.float)

        # Load the connectivity matrix, combine the edges for the specified
        # network type
        same_busbar_e, other_busbar_e, line_e = \
            self.matrix_cache.con_matrices[str(raw_dp['cm_index'])][1]
        if self.network_type == GCN.NetworkType.HOMO:
            dp['edges'] = torch.tensor(np.append(same_busbar_e, line_e, axis=1),
                                       device=self.device,
                                       dtype=torch.long)
        elif self.network_type == GCN.NetworkType.HETERO:
            dp['edges'] = {('object', 'line', 'object'):
                               torch.tensor(line_e,
                                            device=self.device,
                                            dtype=torch.long),
                           ('object', 'same_busbar', 'object'):
                               torch.tensor(same_busbar_e,
                                            device=self.device,
                                            dtype=torch.long),
                           ('object', 'other_busbar', 'object'):
                               torch.tensor(other_busbar_e,
                                            device=self.device,
                                            dtype=torch.long)}

        # If the data is not for training, add information used in
        # validation analysis
        if not self.train:
            dp = self.add_val_info(raw_dp, dp)

        return dp

class ProcessDataPointFCNN(ProcessDataPointStrategy):
    """
    Process a datapoint to obtain the information used by, and in the format used by, a FCNN.
    """

    def __init__(self,
                 device: torch.device,
                 train: bool,
                 feature_statistics: dict):
        """
        Parameters
        ----------
        device : torch.device
            The device to set torch tensors to.
        train : bool
            Whether to process the datapoint for training or not. More information is included for validation/testing.
        feature_statistics : dict
            Dictionary with information (mean, std) used to normalize features.
        """
        super().__init__(device, train, feature_statistics)

    def process_datapoint(self, raw_dp: dict):
        """
        Process a single datapoint, from raw_dp to dp, with the information and formatting for a FCNN model.

        Parameters
        ----------
        raw_dp : dict
            The 'raw' datapoint (not fully true; these raw datapoints should be preprocessed already) from
            which information is extracted.

        Returns
        -------
        dp : dict
            The resulting datapoint.
        """
        dp = {}

        # Add the label
        dp = self.add_processed_label(raw_dp, dp)

        # Load the sub info array, which contains info about to which
        # substation each object belongs
        dp['sub_info'] = raw_dp['sub_info']

        # Load, normalize features (including the topology vector), turn them into a single tensor
        fstats = self.feature_statistics
        norm_gen_features = (np.array(raw_dp['gen_features'])
                             - fstats['gen']['mean']) / fstats['gen']['std']
        norm_load_features = (np.array(raw_dp['load_features'])
                              - fstats['load']['mean']) / fstats['load']['std']
        norm_or_features = (np.array(raw_dp['or_features'])
                            - fstats['or']['mean']) / fstats['or']['std']
        norm_ex_features = (np.array(raw_dp['ex_features'])
                            - fstats['ex']['mean']) / fstats['ex']['std']
        topo_vect = raw_dp['topo_vect']
        dp['features'] = torch.tensor(np.concatenate((norm_gen_features.flatten(),
                                                      norm_load_features.flatten(),
                                                      norm_or_features.flatten(),
                                                      norm_ex_features.flatten(),
                                                      topo_vect)),
                                     device=self.device,
                                     dtype=torch.float)

        # If the data is not for training, add information used in
        # validation analysis
        if not self.train:
            dp = self.add_val_info(raw_dp, dp)

        return dp


class ClassWeightAssigner:
    """
    Class for assigning weights to samples based on their class (i.e. action). Us
    """
    def __init__(self,
                 class_counter_path: str,
                 max_adapt_weight: int,
                 min_adapt_weight: int,
                 min_weight_zero):
        """

        Parameters
        ----------
        class_counter_path : str
            Path of the .json file storing the data structure that stores the frequency of each class in the
            entire (train+val+test) dataset.
        max_adapt_weight : int
            The max threshold for transforming the weight of a class, above which classes are assigned a weight of 1.
        min_adapt_weight : int
            The min threshold for transforming the weight of a class, below which classes are assigned a weight of 1 or
             0.
        min_weight_zero : int
            The max threshold, below which classes are assigned a weight of 0. Used to exclude infrequent classes.
        """
        assert max_adapt_weight >= min_adapt_weight, "Max adapt weight cannot be smaller than the min adapt weight."
        assert min_adapt_weight >= min_weight_zero, "Min adapt weight cannot be smaller than min_weight_zero."
        # Open action counter data structure
        with open(class_counter_path, 'r') as file:
            self._class_counter = json.loads(file.read())

        # Save parameters
        self.max_adapt_weight = max_adapt_weight
        self.min_adapt_weight = min_adapt_weight
        self.min_weight_zero = min_weight_zero

        # Compute relevant numbers
        values = self._class_counter.values()
        self.N_datapoints = sum(values)
        self.N_classes = len(values)
        self.max_class_size = max(values)

    def assign(self, class_hash: int) -> float:
        """
        Given an action/class hash, return the class weights.

        Parameters
        ----------
        class_hash : int
            The hash of the class.

        Returns
        -------
        weight : float
            The weight.

        Raises
        ------
        ValueError : When the action hash isn't stored in the action counter, i.e. it is not in the train/val/test
        data.
        """
        class_count = self._class_counter[str(class_hash)]

        if class_count < self.min_weight_zero:
            return 0
        elif self.min_adapt_weight < class_count < self.max_adapt_weight:
            return self._get_adapted_class_weight(class_count)
        else:
            return 1

    def _get_adapted_class_weight(self, class_count: int) -> float:
        """
        Get the adapted class weight for a class given the class count.

        Parameters
        ----------
        class_count : int
            The class count.

        Returns
        -------
        float
            The adapted class weight.
        """
        return (self.N_datapoints/self.N_classes)/class_count

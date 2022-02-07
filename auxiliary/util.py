#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:04:42 2021

@author: matthijs
"""
import numpy as np

import yaml
import os
from typing import List, Sequence, Callable
import json

import training.models


def load_config():
    """
    Loads the config file as dictionary

    Raises
    ------
    exc
        YAML exception encountered by the YAML parsers.

    Returns
    -------
    dict
        The config file.

    """
    with open('config.yaml') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
    config['training']['GCN']['hyperparams']['network_type'] = training.models.GCN.NetworkType(
        config['training']['GCN']['hyperparams']['network_type']
    )

    # Perform some assertions
    for prm, n in [(config['tutor_generated_data']['n_chronics'], 'n_chronics'),
                   (config['rte_case14_realistic']['ts_in_day'], 'ts_in_day'),
                   (config['rte_case14_realistic']['n_subs'], 'n_subs'),
                   (config['training']['settings']['train_log_freq'], 'train_log_freq'),
                   (config['training']['settings']['val_log_freq'], 'val_log_freq'),
                   (config['training']['hyperparams']['n_epoch'], 'n_epoch'),
                   (config['training']['hyperparams']['lr'], 'lr'),
                   (config['training']['hyperparams']['N_node_hidden'], 'N_node_hidden'),
                   (config['training']['hyperparams']['LReLu_neg_slope'], 'LReLu_neg_slope'),
                   (config['training']['hyperparams']['batch_size'], 'batch_size'),
                   (config['training']['hyperparams']['label_smoothing_alpha'], 'label_smoothing_alpha'),
                   (config['training']['hyperparams']['weight_init_std'], 'weight_init_std'),
                   (config['training']['hyperparams']['weight_decay'], 'weight_decay'),
                   (config['training']['hyperparams']['non_sub_label_weight'], 'non_sub_label_weight'),
                   (config['training']['hyperparams']['early_stopping_patience'], 'early_stopping_patience'),
                   (config['training']['hyperparams']['action_frequency_threshold'], 'action_frequency_threshold'),
                   (config['training']['constants']['estimated_train_size'], 'estimated_train_size'),
                   (config['training']['GCN']['hyperparams']['N_GNN_layers'], 'N_GNN_layers'),
                   (config['training']['GCN']['constants']['N_f_gen'], 'N_f_gen'),
                   (config['training']['GCN']['constants']['N_f_load'], 'N_f_load'),
                   (config['training']['GCN']['constants']['N_f_endpoint'], 'N_f_endpoint'),
                   (config['training']['FCNN']['hyperparams']['N_layers'], 'N_layers'),
                   (config['training']['FCNN']['constants']['size_in'], 'size_in'),
                   (config['training']['FCNN']['constants']['size_out'], 'size_out')]:
        assert prm >= 0, f'Parameter {n} should not be negative.'
    assert all(l >= 0 for l in config['tutor_generated_data']['line_idxs_to_consider_N-1']), \
        "Line idx cannot be negative."
    assert all(l >= 0 for l in config['rte_case14_realistic']['thermal_limits']), \
        "Thermal limit cannot be negative."
    assert (max(config['tutor_generated_data']['line_idxs_to_consider_N-1']) + 1 <=
            len(config['rte_case14_realistic']['thermal_limits'])), "Line idx plus one cannot be higher than" + \
           " the number of lines."
    assert 0 <= config['dataset']['train_perc'] <= 1, "Train. perc. should be in percentage range."
    assert 0 <= config['dataset']['val_perc'] <= 1, "Val. perc. should be in percentage range."
    assert config['training']['hyperparams']['model_type'] in ['GGN', 'FCNN'], \
           "Model_type should be value GCN or FCNN."
    assert config['training']['GCN']['hyperparams']['aggr'] in ['add', 'mean'], \
           "Aggr. should be mean or add."

    return config


def set_wd_to_package_root():
    """
    Set the working directory to the root of the package.
    """
    os.chdir(os.path.dirname(__file__) + '/..')


def flatten(t: Sequence[Sequence]) -> List:
    """
    Flatten a sequence of sequences.

    Parameters
    ----------
    t : Sequence[Sequence]
        The sequence of sequences to flatten.

    Returns
    -------
    List
        Flattened sequence.
    """
    return [item for sublist in t for item in sublist]


def hash_nparray(arr: np.array) -> int:
    """
    Hashes a numpy array.

    Parameters
    ----------
    arr : np.array
        The array.

    Returns
    -------
    int
        The hash value.
    """
    return hash(arr.data.tobytes())


class NumpyEncoder(json.JSONEncoder):
    """
    Class that can be used in json.dump() to encode np.array objects.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def argmax_f(s: Sequence, f: Callable) -> int:
    """
    Take the argmax (i.e. the index) based on the maximum of a particular
    function.

    Parameters
    ----------
    s : Sequence
        The sequence to find the argmax of.
    f : Callable
        The function to apply to the elements

    Returns
    -------
    int
        The index produced by the argmax.

    """
    return max(enumerate([f(d) for d in s]),
               key=lambda x: x[1])[0]

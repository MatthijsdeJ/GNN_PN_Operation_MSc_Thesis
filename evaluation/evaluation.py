# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:51:31 2022

@author: matthijs
"""

import grid2op
import auxiliary.grid2op_util as g2o_util
from auxiliary.grid2op_util import ts_to_day, select_single_substation_from_topovect, env_step_raise_exception
from auxiliary.config import get_config, StrategyType, ModelType
import torch
import evaluation.strategy as strat
from training.models import GCN, FCNN, Model
import json
import logging
from auxiliary.generate_action_space import get_env_actions
import numpy as np
import os
from typing import List, Dict


def evaluate():
    """
    Generate imitation learning data from the tutor model.
    """
    # Load constants, settings, hyperparameters, arguments
    config = get_config()
    num_chronics = config['evaluation']['n_chronics']
    ts_in_day = int(config['rte_case14_realistic']['ts_in_day'])
    disable_line = config['evaluation']['disable_line']
    logging_path = config['paths']['evaluation_log']
    save_data = config['evaluation']['save_data']

    # Initialize logging
    logging.basicConfig(filename=logging_path, filemode='w', format='%(message)s', level=logging.INFO)

    # Initialize environment
    env = g2o_util.init_env(grid2op.Rules.AlwaysLegal)
    print("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))

    # Initialize strategy
    strategy = init_strategy(env)

    # Loop over chronics
    for num in range(0, num_chronics):
        try:
            print('current chronic: %s' % env.chronics_handler.get_name())
            logging.info('current chronic: %s' % env.chronics_handler.get_name())

            # (Re)set variables
            days_completed = 0
            if save_data:
                chronic_datapoints = day_datapoints = []

            # Disable lines, if any
            if disable_line != -1:
                obs = env_step_raise_exception(env, env.action_space({"set_line_status": (disable_line, -1)}))
            else:
                obs = env_step_raise_exception(env, env.action_space())

            # Save reference topology
            reference_topo_vect = obs.topo_vect.copy()

            # Loop over timesteps until exhausted
            while env.nb_time_step < env.chronics_handler.max_timestep():

                # Reset at midnight
                if env.nb_time_step % ts_in_day == ts_in_day - 1:
                    print(f'Day {ts_to_day(env.nb_time_step, ts_in_day)} completed.')
                    logging.info(f'Day {ts_to_day(env.nb_time_step, ts_in_day)} completed.')

                    days_completed += 1

                    # Reset topology
                    obs = env_step_raise_exception(env, env.action_space({'set_bus': reference_topo_vect}))

                    if save_data:
                        # Save and reset data
                        chronic_datapoints += day_datapoints
                        day_datapoints = []

                    continue

                # Strategy selects an action
                obs = env.get_obs()
                if not save_data:
                    action, _ = strategy.select_action(obs)
                else:
                    action, datapoint = strategy.select_action(obs)

                # Take the selected action in the environment
                previous_max_rho = obs.rho.max()
                previous_topo_vect = obs.topo_vect
                obs = env_step_raise_exception(env, action)

                # Potentially log action information
                if previous_max_rho > config['evaluation']['activity_threshold']:
                    mask, sub_id = select_single_substation_from_topovect(torch.tensor(action.set_bus),
                                                                          torch.tensor(obs.sub_info),
                                                                          select_nothing_condition=lambda x:
                                                                          not any(x) or x == previous_topo_vect)
                    msg = "Old max rho, new max rho, substation, set_bus: " + \
                          str((previous_max_rho, obs.rho.max(), sub_id, list(action.set_bus[mask == 1])))
                    print(msg)
                    logging.info(msg)

                # Save action data
                if save_data and datapoint is not None:
                    day_datapoints.append(datapoint)

                # If the game is done at this point, this indicated a (failed) game over.
                # If so, reset the environment to the start of next day and discard the records
                if env.done:
                    print(f'Failure at step {env.nb_time_step} on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                    logging.info(f'Failure at step {env.nb_time_step} on day {ts_to_day(env.nb_time_step, ts_in_day)}')

                    g2o_util.skip_to_next_day(env, ts_in_day, int(env.chronics_handler.get_name()), disable_line)
                    day_datapoints = []

            # At the end of a chronic, print a message, and store and reset the corresponding records
            print('Chronic exhausted! \n\n\n')
            logging.info('Chronic exhausted! \n\n\n')

            # Saving and resetting the data
            if save_data:
                save_records(chronic_datapoints, int(env.chronics_handler.get_name()), days_completed)
        except grid2op.Exceptions.DivergingPowerFlow:
            msg = (f'Diverging-powerflow exception at step {env.nb_time_step} on '
                   f'day {ts_to_day(env.nb_time_step, ts_in_day)}. Skipping this chronic.')
            print(msg)
            logging.info(msg)
        finally:
            env.reset()


def init_model() -> Model:
    """
    Initialize the machine learning model.

    Returns
    -------
    model : Model
        The machine learning model.
    """
    config = get_config()
    train_config = config['training']
    model_path = config['paths']['model']

    # Initialize model
    if train_config['hyperparams']['model_type'] == ModelType.GCN:
        model = GCN(train_config['hyperparams']['LReLu_neg_slope'],
                    train_config['hyperparams']['weight_init_std'],
                    train_config['GCN']['constants']['N_f_gen'],
                    train_config['GCN']['constants']['N_f_load'],
                    train_config['GCN']['constants']['N_f_endpoint'],
                    train_config['GCN']['hyperparams']['N_GCN_layers'],
                    train_config['hyperparams']['N_node_hidden'],
                    train_config['GCN']['hyperparams']['aggr'],
                    train_config['GCN']['hyperparams']['network_type'],
                    train_config['GCN']['hyperparams']['layer_type'],
                    train_config['GCN']['hyperparams']['GINConv_nn_depth'])
    elif train_config['hyperparams']['model_type'] == ModelType.FCNN:
        model = FCNN(train_config['hyperparams']['LReLu_neg_slope'],
                     train_config['hyperparams']['weight_init_std'],
                     train_config['FCNN']['constants']['size_in'],
                     train_config['FCNN']['constants']['size_out'],
                     train_config['FCNN']['hyperparams']['N_layers'],
                     train_config['hyperparams']['N_node_hidden'])
    else:
        raise ValueError("Invalid model_type value.")

    # Load model
    model.load_state_dict(torch.load(model_path))

    return model


def init_strategy(env: grid2op.Environment) -> strat.AgentStrategy:
    """
    Initialize the strategy.

    Parameters
    ----------
    env : grid2op.Environment
        The grid2op environment.

    Returns
    -------
    strategy : StrategyType
        The initialized strategy.
    """
    config = get_config()
    strategy_type = config['evaluation']['strategy']

    if strategy_type == StrategyType.IDLE:
        strategy = strat.IdleStrategy(env.action_space({}))
    elif strategy_type == StrategyType.GREEDY:
        strategy = strat.GreedyStrategy(config['evaluation']['activity_threshold'],
                                        env.action_space({}),
                                        get_env_actions(env, disable_line=config['evaluation']['disable_line']))
    elif strategy_type == StrategyType.N_MINUS_ONE:
        strategy = strat.NMinusOneStrategy(config['evaluation']['activity_threshold'],
                                           env.action_space,
                                           get_env_actions(env, disable_line=config['evaluation']['disable_line']),
                                           config['evaluation']['NMinusOne_strategy']['line_idxs_to_consider_N-1'],
                                           config['evaluation']['NMinusOne_strategy']['N0_rho_threshold'])
    elif strategy_type == StrategyType.NAIVE_ML:
        # Initialize model and normalization statistics
        model = init_model()
        feature_statistics_path = config['paths']['feature_statistics']
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        # Initialize strategy
        strategy = strat.NaiveStrategy(model,
                                       feature_statistics,
                                       env.action_space,
                                       config['evaluation']['activity_threshold'])
    elif strategy_type == StrategyType.VERIFY_ML:
        # Initialize model and normalization statistics
        model = init_model()
        feature_statistics_path = config['paths']['feature_statistics']
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        # Initialize strategy
        strategy = strat.VerifyStrategy(model,
                                        feature_statistics,
                                        env.action_space,
                                        config['evaluation']['activity_threshold'],
                                        config['evaluation']['verify_strategy']['reject_action_threshold'])
    elif strategy_type == StrategyType.VERIFY_GREEDY_HYBRID:
        # Initialize model and normalization statistics
        model = init_model()
        feature_statistics_path = config['paths']['feature_statistics']
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        # Initialize strategy
        strategy = strat.VerifyGreedyHybridStrategy(model,
                                                    feature_statistics,
                                                    env.action_space,
                                                    config['evaluation']['activity_threshold'],
                                                    config['evaluation']['verify_strategy']['reject_action_threshold'],
                                                    get_env_actions(env,
                                                                    disable_line=config['evaluation']['disable_line']),
                                                    config['evaluation']['hybrid_strategies'][
                                                        'take_the_wheel_threshold'])
    elif strategy_type == StrategyType.VERIFY_N_MINUS_ONE_HYBRID:
        # Initialize model and normalization statistics
        model = init_model()
        feature_statistics_path = config['paths']['feature_statistics']
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        # Initialize strategy
        strategy = strat.VerifyNMinusOneHybridStrategy(model,
                                                       feature_statistics,
                                                       env.action_space,
                                                       config['evaluation']['activity_threshold'],
                                                       config['evaluation']['verify_strategy'][
                                                           'reject_action_threshold'],
                                                       get_env_actions(env, disable_line=config['evaluation'][
                                                           'disable_line']),
                                                       config['evaluation']['NMinusOne_strategy'][
                                                           'line_idxs_to_consider_N-1'],
                                                       config['evaluation']['NMinusOne_strategy']['N0_rho_threshold'],
                                                       config['evaluation']['hybrid_strategies'][
                                                            'take_the_wheel_threshold'])
    else:
        raise ValueError("Invalid value for strategy_name.")

    return strategy


def save_records(datapoints: List[Dict],
                 chronic: int,
                 days_completed: int):
    """
    Saves records of a chronic to disk and prints a message that they are saved.

    Parameters
    ----------
    datapoints : list[Dict]
        The recorded datapoints.
    chronic : int
        Integer representing the chronic which is saved.
    days_completed : int
        The number of days completed.
    """
    config = get_config()
    save_path = config['paths']['tutor_imitation']
    do_nothing_capacity_threshold = config['evaluation']['activity_threshold']
    lout = config['evaluation']['disable_line']

    if datapoints:
        dp_matrix = np.zeros((0, 5 + len(datapoints[0]['observation_vector'])), dtype=np.float32)
        for dp in datapoints:
            dp_vector = np.concatenate(([dp['action_index'], dp['do_nothing_rho'], dp['action_rho'], dp['duration'],
                                         dp['timestep']],
                                        dp['observation_vector']))
            dp_vector = np.reshape(dp_vector, (1, -1)).astype(np.float32)
            dp_matrix = np.concatenate((dp_matrix, dp_vector), axis=0)
    else:
        dp_matrix = np.array()

    folder_name = f'records_chronics_lout:{lout}_dnthreshold:{do_nothing_capacity_threshold}'
    file_name = f'records_chronic:{chronic}_dayscomp:{days_completed}.npy'
    if not os.path.isdir(os.path.join(save_path, folder_name)):
        os.mkdir(os.path.join(save_path, folder_name))
    np.save(os.path.join(save_path, folder_name, file_name), dp_matrix)
    print('# records are saved! #')

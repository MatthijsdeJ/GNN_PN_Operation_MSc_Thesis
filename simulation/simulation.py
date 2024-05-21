# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:51:31 2022

@author: matthijs
"""
import random

import grid2op
import auxiliary.grid2op_util as g2o_util
from auxiliary.grid2op_util import ts_to_day, select_single_substation_from_topovect, env_step_raise_exception
from auxiliary.config import get_config, StrategyType, ModelType
import torch
import simulation.strategy as strat
from training.models import GCN, FCNN, Model
import json
import logging
from auxiliary.generate_action_space import get_env_actions
import numpy as np
import os
from typing import List, Dict
import time


def simulate():
    """
    Generate imitation learning data from the tutor model.
    """
    # Load constants, settings, hyperparameters, arguments
    global config
    global ts_in_day

    config = get_config()
    n_chronics = config['simulation']['n_chronics']
    partition = config['simulation']['partition']
    ts_in_day = int(config['rte_case14_realistic']['ts_in_day'])
    disable_line = config['simulation']['disable_line']
    logging_path = config['paths']['evaluation_log']
    save_data = config['simulation']['save_data']

    # Initialize logging
    logging.basicConfig(filename=logging_path, filemode='w', format='%(message)s', level=logging.INFO)

    # Initialize environment
    env = g2o_util.init_env()

    # Initialize strategy
    strategy = init_strategy(env)

    # Specify scenarios
    if partition == 'train':
        scenarios = np.load(config['paths']['data_split'] + 'train_scenarios.npy')
    elif partition == 'val':
        scenarios = np.load(config['paths']['data_split'] + 'val_scenarios.npy')
    elif partition == 'test':
        scenarios = np.load(config['paths']['data_split'] + 'test_scenarios.npy')
    elif partition == 'all':
        scenarios = range(0, n_chronics)
    else:
        raise ValueError()

    # Log config
    log_and_print(f'Config: {config}')

    # Loop over chronics
    for num in scenarios:
        env.set_id(num)
        env.reset()

        try:
            log_and_print(f'{env.nb_time_step}: Current chronic: %s' % env.chronics_handler.get_name())

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

            # Capture time for analysing durations
            start_day_time = time.thread_time_ns() / 1e9

            # Create opponent action
            attack1_begin, attack1_end, attack1_line, attack2_begin, attack2_end, attack2_line \
                = _create_opponent_variables()

            # While chronic is not completed
            while env.nb_time_step < env.chronics_handler.max_timestep():

                # Reset at midnight,  add day data to chronic data
                if env.nb_time_step % ts_in_day == ts_in_day - 1:

                    end_day_time = time.thread_time_ns() / 1e9

                    log_and_print(f'{env.nb_time_step}: Day {ts_to_day(env.nb_time_step, ts_in_day)} completed in '
                                  f'{end_day_time - start_day_time:.2f} seconds.')
                    days_completed += 1
                    start_day_time = time.thread_time_ns() / 1e9

                    # Reset topology
                    env_step_raise_exception(env, env.action_space({'set_bus': reference_topo_vect}))

                    # Reset opponent
                    attack1_begin, attack1_end, attack1_line, attack2_begin, attack2_end, attack2_line \
                        = _create_opponent_variables(env.nb_time_step)

                    # Save and reset data
                    if save_data:
                        chronic_datapoints += day_datapoints
                        day_datapoints = []

                    continue

                # Strategy selects an action
                obs = env.get_obs()
                before_action_time = time.thread_time_ns() / 1e3
                action, datapoint = strategy.select_action(obs)
                action_duration = time.thread_time_ns() / 1e3 - before_action_time

                # Assert not more than one substation is changed and no lines are changed
                assert (action._subs_impacted is None) or (sum(action._subs_impacted) < 2), \
                    ("Actions should at most impact a single substation.")
                assert np.array_equal(obs.line_status, (obs + action).line_status), \
                    ("Action should not impact the line status.")

                timestep = env.nb_time_step

                # Disable lines
                if env.nb_time_step in [attack1_begin, attack2_begin]:
                    attack_line = attack1_line if env.nb_time_step == attack1_begin else attack2_line
                    line_status_copy = action.set_line_status.copy()
                    line_status_copy[attack_line] = -1
                    action.set_line_status = line_status_copy
                    log_and_print(f"{env.nb_time_step}: Line {attack_line} disabled by attack.")

                # Assert check disabled lines
                if attack1_begin < timestep < attack1_end:
                    assert obs.line_status[attack1_line] == False
                if attack2_begin < timestep < attack2_end:
                    assert obs.line_status[attack2_line] == False

                # Re-enable lines
                if env.nb_time_step in [attack1_end, attack2_end]:
                    attack_line = attack1_line if env.nb_time_step == attack1_end else attack2_line
                    line_status_copy = action.set_line_status.copy()
                    line_status_copy[attack_line] = 1
                    action.set_line_status = line_status_copy
                    log_and_print(f"{env.nb_time_step}: Line {attack_line} no longer disabled by attack.")

                # Take the selected action in the environment
                previous_max_rho = obs.rho.max()
                previous_topo_vect = obs.topo_vect
                obs, _, _, _ = env.step(action)

                # Potentially log action information
                if previous_max_rho > config['simulation']['activity_threshold']: #and not env.done:
                    topo_vect_diff = 1 - np.equal(previous_topo_vect, obs.topo_vect)
                    mask, sub_id = select_single_substation_from_topovect(torch.tensor(topo_vect_diff),
                                                                          torch.tensor(obs.sub_info),
                                                                          select_nothing_condition=lambda x:
                                                                          not any(x)
                                                                          )
                    log_and_print(f"{env.nb_time_step}: Action selected. "
                                  f"Old max rho: {previous_max_rho:.4f}, "
                                  f"new max rho: {obs.rho.max():.4f}, "
                                  f"substation: {sub_id}, "
                                  f"configuration: {list(obs.topo_vect[mask == 1])}, "
                                  f"action duration in microsecond: {int(action_duration)}.")

                # Save action data
                if save_data and datapoint is not None:
                    day_datapoints.append(datapoint)

                # If the game is done at this point, this indicated a (failed) game over.
                # If so, reset the environment to the start of next day and discard the records
                if env.done:
                    log_and_print(f'{env.nb_time_step}: Failure of day {ts_to_day(env.nb_time_step, ts_in_day)}.')
                    g2o_util.skip_to_next_day(env, ts_in_day, int(env.chronics_handler.get_name()), disable_line)
                    day_datapoints = []
                    start_day_time = time.thread_time_ns() / 1e9
                    # Reset opponent
                    attack1_begin, attack1_end, attack1_line, attack2_begin, attack2_end, attack2_line \
                        = _create_opponent_variables(env.nb_time_step)

            # At the end of a chronic, print a message, and store and reset the corresponding records
            log_and_print(f'{env.nb_time_step}: Chronic exhausted! \n\n\n')

            # Saving and resetting the data
            if save_data:
                save_records(chronic_datapoints, int(env.chronics_handler.get_name()), days_completed)

        except (grid2op.Exceptions.DivergingPowerFlow, grid2op.Exceptions.BackendError) as e:
            log_and_print(f'{env.nb_time_step}: Uncaught exception encountered on ' +
                          f'day {ts_to_day(env.nb_time_step, ts_in_day)}: {e}.' +
                          ("" if not hasattr(e, '__notes__') else " ".join(e.__notes__)) +
                          ". Skipping this scenario. \n\n\n")


def _create_opponent_variables(day_offset: int = 0):
    """
    Create the opponent variables:
    attack1_begin, attack1_end, attack1_line, attack2_begin, attack2_end, attack2_line


    Parameters
    ----------
    day_offset : int
        The timestep that denotes the start of the day.

    Returns
    -------
    attack1_begin : int
        The timestep where the first attack starts.
    attack1_end : int
        The timestep where the first attack ends.
    attack1_line : int
        The line disabled by the first attack.
    attack2_begin : int
        The timestep where the second attack ends.
    attack2_end : int
        The timestep where the second attack ends.
    attack2_line : int
        The line disabled by the second attack.
    """
    config = get_config()
    attack_lines = config['simulation']['opponent']['attack_lines']
    attack_duration = config['simulation']['opponent']['attack_duration']
    attack_cooldown = config['simulation']['opponent']['attack_cooldown']

    attack1_begin = min(random.randint(1, ts_in_day - 2 * attack_duration - attack_cooldown - 2),
                        random.randint(1, ts_in_day - 2 * attack_duration - attack_cooldown - 2))
    attack1_end = attack1_begin + attack_duration
    attack1_line = random.choice(attack_lines)

    attack2_begin = random.randint(attack1_end + attack_cooldown, ts_in_day - attack_duration - 1)
    attack2_end = attack2_begin + attack_duration
    attack2_line = random.choice(attack_lines)

    attack1_begin += day_offset
    attack1_end += day_offset
    attack2_begin += day_offset
    attack2_end += day_offset

    return attack1_begin, attack1_end, attack1_line, attack2_begin, attack2_end, attack2_line


def log_and_print(msg: str):
    """
    Log and print a message.

    Parameters
    ----------
    msg : str
        The message.
    """
    print(msg)
    logging.info(msg)


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
    strategy_type = config['simulation']['strategy']

    if strategy_type == StrategyType.IDLE:
        strategy = strat.IdleStrategy(env.action_space({}))
    elif strategy_type == StrategyType.GREEDY:
        strategy = strat.GreedyStrategy(config['simulation']['activity_threshold'],
                                        env.action_space({}),
                                        get_env_actions(env, disable_line=config['simulation']['disable_line']))
    elif strategy_type == StrategyType.N_MINUS_ONE:
        strategy = strat.NMinusOneStrategy(config['simulation']['activity_threshold'],
                                           env.action_space,
                                           get_env_actions(env, disable_line=config['simulation']['disable_line']),
                                           config['simulation']['NMinusOne_strategy']['line_idxs_to_consider_N-1'],
                                           config['simulation']['NMinusOne_strategy']['N0_rho_threshold'])
    elif strategy_type == StrategyType.NAIVE_ML:
        # Initialize model and normalization statistics
        model = init_model()
        feature_statistics_path = config['paths']['data']['processed'] + 'auxiliary_data_objects/feature_stats.json'
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        # Initialize strategy
        strategy = strat.NaiveStrategy(model,
                                       feature_statistics,
                                       env.action_space,
                                       config['simulation']['activity_threshold'])
    elif strategy_type == StrategyType.VERIFY_ML:
        # Initialize model and normalization statistics
        model = init_model()
        feature_statistics_path = config['paths']['data']['processed'] + 'auxiliary_data_objects/feature_stats.json'
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())

        # Initialize strategy
        strategy = strat.VerifyStrategy(model,
                                        feature_statistics,
                                        env.action_space,
                                        config['simulation']['activity_threshold'],
                                        config['simulation']['verify_strategy']['reject_action_threshold'])
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
                                                    config['simulation']['activity_threshold'],
                                                    config['simulation']['verify_strategy']['reject_action_threshold'],
                                                    get_env_actions(env,
                                                                    disable_line=config['simulation']['disable_line']),
                                                    config['simulation']['hybrid_strategies'][
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
                                                       config['simulation']['activity_threshold'],
                                                       config['simulation']['verify_strategy'][
                                                           'reject_action_threshold'],
                                                       get_env_actions(env, disable_line=config['simulation'][
                                                           'disable_line']),
                                                       config['simulation']['NMinusOne_strategy'][
                                                           'line_idxs_to_consider_N-1'],
                                                       config['simulation']['NMinusOne_strategy']['N0_rho_threshold'],
                                                       config['simulation']['hybrid_strategies'][
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
    do_nothing_capacity_threshold = config['simulation']['activity_threshold']
    lout = config['simulation']['disable_line']

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

    folder_name = f'records_chronics_lout_{lout}_dnthreshold_{do_nothing_capacity_threshold}'
    file_name = f'records_chronic_{chronic}_dayscomp_{days_completed}.npy'
    if not os.path.isdir(os.path.join(save_path, folder_name)):
        os.mkdir(os.path.join(save_path, folder_name))
    np.save(os.path.join(save_path, folder_name, file_name), dp_matrix)
    print('# records are saved! #')


def env_step_evaluate_exceptions(env: grid2op.Environment.Environment, action: grid2op.Action.BaseAction) \
        -> grid2op.Observation.CompleteObservation:
    obs, _, done, info = env.step(action)

    if len(info['exception']) > 1:
        raise ExceptionGroup('Exceptions', info['exception'])
    elif len(info['exception']) == 1:
        exception = info['exception'][0]
        exception.add_note("\nInfo: " + str(info).strip())
        raise exception

    return obs

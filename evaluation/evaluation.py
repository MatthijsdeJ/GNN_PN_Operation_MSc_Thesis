#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:51:31 2022

@author: matthijs
"""

import grid2op
import auxiliary.grid2op_util as g2o_util
from auxiliary.grid2op_util import ts_to_day
from auxiliary.config import get_config, StrategyType, ModelType
import torch
from evaluation.strategy import NaiveStrategy
from training.models import GCN, FCNN
import json
import logging

def evaluate():
    """
    Generate imitation learning data from the tutor model.
    """
    # Load constants, settings, hyperparameters, arguments
    config = get_config()
    num_chronics = config['tutor_generated_data']['n_chronics']
    ts_in_day = int(config['rte_case14_realistic']['ts_in_day'])
    disable_line = config['evaluation']['disable_line']
    feature_statistics_path = config['paths']['feature_statistics']
    logging_path = config['paths']['evaluation_log']

    # Initialize logging
    logging.basicConfig(filename=logging_path,
                        filemode='w',
                        format='%(message)s',
                        level=logging.INFO)

    # Initialize environment
    env = g2o_util.init_env(grid2op.Rules.AlwaysLegal)
    print("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))

    # Initialize model and normalization statistics
    model = init_model()
    with open(feature_statistics_path, 'r') as file:
        feature_statistics = json.loads(file.read())

    # Initialize strategy
    strategy_type = config['evaluation']['strategy']
    if strategy_type == StrategyType.NAIVE:
        strategy = NaiveStrategy(model,
                                 feature_statistics,
                                 env.action_space,
                                 config['evaluation']['activity_threshold'])
    else:
        raise ValueError("Invalid value for strategy_name.")

    # Loop over chronics
    for num in range(0, num_chronics):

        # (Re)set variables
        obs = env.reset()
        days_completed = 0
        fast_forward_divergingpowerflow_exception = False
        print('current chronic: %s' % env.chronics_handler.get_name())
        logging.info('current chronic: %s' % env.chronics_handler.get_name())

        # Disable lines, if any
        if disable_line != -1:
            obs, _, _, _ = env.step(env.action_space({"set_line_status": (disable_line, -1)}))

        # Save reference topology
        reference_topo_vect = obs.topo_vect.copy()

        # Loop over timesteps until exhausted
        while env.nb_time_step < env.chronics_handler.max_timestep():
            # Sporadically, when fast-forwarding, a diverging powerflow exception can occur.  If that exception
            # has occurred, we skip to the next day.
            if fast_forward_divergingpowerflow_exception:
                print(f'Powerflow exception at step {env.nb_time_step} ' +
                      f'on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                logging.error(f'Powerflow exception at step {env.nb_time_step} ' +
                              f'on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                info = g2o_util.skip_to_next_day(env, ts_in_day,
                                                 num, disable_line)
                continue
                
            # At midnight, reset the topology to the reference, store days' records, reset days' records
            if env.nb_time_step % ts_in_day == ts_in_day-1:
                print(f'Day {ts_to_day(env.nb_time_step, ts_in_day)} completed.')
                logging.info(f'Day {ts_to_day(env.nb_time_step, ts_in_day)} completed.')
                obs, _, _, _ = env.step(env.action_space({'set_bus':
                                                          reference_topo_vect}))
                days_completed += 1
                continue
                
            # If neither of above holds, the model takes an action
            obs = env.get_obs()
            action = strategy.select_action(obs)

            # Take the selected action in the environment
            old_rho = obs.rho.max()
            old_tv = obs.topo_vect
            obs, _, _, _ = env.step(action)

            # Potentially log action information
            if old_rho > config['evaluation']['activity_threshold']:
                mask, sub_id = g2o_util.select_single_substation_from_topovect(torch.tensor(action.set_bus != old_tv),
                                                                               torch.tensor(obs.sub_info),
                                                                               select_nothing_condition= lambda x:
                                                                               not any(x))
                msg = "Old max rho, new max rho, substation, set_bus: " + \
                      str((old_rho,  obs.rho.max(),  sub_id,  list(action.set_bus[mask])))
                print(msg)
                logging.info(msg)

            # If the game is done at this point, this indicated a (failed) game over.
            # If so, reset the environment to the start of next day and discard the records
            if env.done:
                print(f'Failure at step {env.nb_time_step} on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                logging.info(f'Failure at step {env.nb_time_step} on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                fast_forward_divergingpowerflow_exception = g2o_util.skip_to_next_day(env,  ts_in_day,
                                                                                      num, disable_line)

        # At the end of a chronic, print a message, and store and reset the corresponding records
        print('Chronic exhausted! \n\n\n')
        logging.info('Chronic exhausted! \n\n\n')


def init_model():
    """

    Returns
    -------

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

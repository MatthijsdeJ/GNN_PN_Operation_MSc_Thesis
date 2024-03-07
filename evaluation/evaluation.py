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
import evaluation.strategy as strat
from training.models import GCN, FCNN, Model
import json
import logging
from auxiliary.generate_action_space import get_env_actions

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
    strategy = init_strategy(env, model, feature_statistics)

    # Loop over chronics
    for num in range(0, num_chronics):

        # (Re)set variables
        print('current chronic: %s' % env.chronics_handler.get_name())
        logging.info('current chronic: %s' % env.chronics_handler.get_name())
        divergingpowerflow_exception = False

        # Disable lines, if any
        if disable_line != -1:
            obs, _, _, _ = env.step(env.action_space({"set_line_status": (disable_line, -1)}))
        else:
            obs, _, _, _ = env.step(env.action_space({}))

        # Save reference topology
        reference_topo_vect = obs.topo_vect.copy()

        # Loop over timesteps until exhausted
        while env.nb_time_step < env.chronics_handler.max_timestep():
            # Sporadically, when fast-forwarding, a diverging powerflow exception can occur.  If that exception
            # has occurred, we skip to the next day.
            if divergingpowerflow_exception:
                print(f'Powerflow exception at step {env.nb_time_step} ' +
                      f'on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                logging.error(f'Powerflow exception at step {env.nb_time_step} ' +
                              f'on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                _ = g2o_util.skip_to_next_day(env, ts_in_day, num, disable_line)
                continue
                
            # At midnight, reset the topology to the reference, store days' records, reset days' records
            if env.nb_time_step % ts_in_day == ts_in_day-1:
                print(f'Day {ts_to_day(env.nb_time_step, ts_in_day)} completed.')
                logging.info(f'Day {ts_to_day(env.nb_time_step, ts_in_day)} completed.')
                obs, _, _, _ = env.step(env.action_space({'set_bus':
                                                          reference_topo_vect}))
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
                mask, sub_id = g2o_util.select_single_substation_from_topovect(torch.tensor(action.set_bus),
                                                                               torch.tensor(obs.sub_info),
                                                                               select_nothing_condition=lambda x:
                                                                               not any(x) or x == old_tv)
                msg = "Old max rho, new max rho, substation, set_bus: " + \
                      str((old_rho,  obs.rho.max(),  sub_id,  list(action.set_bus[mask == 1])))
                print(msg)
                logging.info(msg)

            # If the game is done at this point, this indicated a (failed) game over.
            # If so, reset the environment to the start of next day and discard the records
            if env.done:
                print(f'Failure at step {env.nb_time_step} on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                logging.info(f'Failure at step {env.nb_time_step} on day {ts_to_day(env.nb_time_step, ts_in_day)}')
                divergingpowerflow_exception = g2o_util.skip_to_next_day(env,
                                                                         ts_in_day,
                                                                         int(env.chronics_handler.get_name()),
                                                                         disable_line)

        # At the end of a chronic, print a message, and store and reset the corresponding records
        print('Chronic exhausted! \n\n\n')
        logging.info('Chronic exhausted! \n\n\n')

        # Resetting environment
        env.reset()
        divergingpowerflow_exception = False


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


def init_strategy(env: grid2op.Environment, model: Model, feature_statistics: dict) -> strat.AgentStrategy:
    """
    Initialize the strategy.

    Parameters
    ----------
    env : grid2op.Environment
        The grid2op environment.
    model : Model
        The machine learning model.
    feature_statistics : dict[dict[float]]
        The statistics (meand, stds) per object type (gen, load, or, ex).


    Returns
    -------
    strategy : StrategyType
        The initialized strategy.
    """
    config = get_config()
    strategy_type = config['evaluation']['strategy']

    if strategy_type == StrategyType.IDLE:
        strategy = strat.IdleStrategy(env.action_space)
    elif strategy_type == StrategyType.NAIVE:
        strategy = strat.NaiveStrategy(model,
                                       feature_statistics,
                                       env.action_space,
                                       config['evaluation']['activity_threshold'])
    elif strategy_type == StrategyType.VERIFY:
        strategy = strat.VerifyStrategy(model,
                                        feature_statistics,
                                        env.action_space,
                                        config['evaluation']['activity_threshold'],
                                        config['evaluation']['verify_strategy']['reject_action_threshold'])
    elif strategy_type == StrategyType.HYBRID:
        strategy = strat.HybridStrategy(model,
                                        feature_statistics,
                                        env.action_space,
                                        config['evaluation']['activity_threshold'],
                                        config['evaluation']['hybrid_strategy']['reject_action_threshold'],
                                        config['evaluation']['hybrid_strategy']['greedy_take_the_wheel_threshold'],
                                        get_env_actions(env, disable_line=config['evaluation']['disable_line']))
    else:
        raise ValueError("Invalid value for strategy_name.")

    return strategy

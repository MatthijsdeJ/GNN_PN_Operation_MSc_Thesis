"""
ADAPTED FROM:
https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution/blob/master/Tutor/Tutor.py

Protected by Mozilla Public License Version 2.0.

In this file, we feed Tutor with numerous scenarios, and obtain a teaching
dataset in form of (feature: observation, label: action chosen).
The dataset is used for imitation learning of Junior Student afterward.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import grid2op
import numpy as np
from Tutor.Tutor import Tutor
from action_space.generate_action_space import get_env_actions
import datetime as dt
import argparse
import util

# =============================================================================
# An alternative way to return to the reference topology is to simply take
# regular actions to get there. However, this is not trivial, as the most
# straightforward path towards the reference topology might be 'unsafe'
# (i.e. cause overloading). 
#
# The function below still requires testing and is hence commented out.
#
# def return_to_reference(reference_topo_vect, obs, env):
#     s=0
#     done=False
#     for i in obs.sub_info:
#         if not np.array_equal(reference_topo_vect[s:i], obs.topo_vect[s:i]):
#             act = env.action_space({'set_bus': reference_topo_vect[s:i]})
#             obs, _, done, info = env.step(act)
#             print(info)
#             exit()
#             
#             if done:
#                 break
#         s+=i
#     
#     return obs, done
# =============================================================================

def init_env(config: dict) ->  grid2op.Environment.Environment:
    '''
    Prepares the Grid2Op environment from a dictionary containing configuration
    setting.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration variables.

    Returns
    -------
    env : TYPE
        The Grid2Op environment.

    '''
    data_path = config['paths']['rte_case14_realistic']
    scenario_path = config['paths']['rte_case14_realistic_chronics']

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=data_path, chronics_path=scenario_path, backend=backend,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
    except:
        env = grid2op.make(dataset=data_path, chronics_path=scenario_path,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
        
    # for reproducible experiments
    env.seed(config['tutor_generated_data']['seed'])  

    #Set custom thermal limits
    thermal_limits = config['rte_case14_realistic']['thermal_limits']
    env.set_thermal_limit(thermal_limits)
    return env
    
    
def save_records(records: np.array, chronic: int, save_path: str,
                 do_nothing_capacity_threshold: float, lout: int = -1,):
    '''
    Saves records of a chronic to disk and prints a message that they are saved. 

    Parameters
    ----------
    records : np.array
        The records.
    chronic : int
        Int representing the chronic which is saved.
    save_path : str
        Path where the output folder with the records file is to be made.
    do_nothing_capacity_threshold : int
        The threshold max. line rho at which the tutor takes actions.
    lout : int
        Index of any line that is out.
    '''

    
    folder_name = f'records_chronics_lout:{lout}_dnthreshold:{do_nothing_capacity_threshold}'
    file_name = f'records_chronic:{chronic}.npy'
    if not os.path.isdir(os.path.join(save_path,folder_name)):
        os.mkdir(os.path.join(save_path,folder_name))
    np.save(os.path.join(save_path,folder_name,file_name), records)
    print('# records are saved! #')
    

def empty_records(obs_vect_size: int):
    '''
    Generate a numpy array for storing records of actions and observations.

    Parameters
    ----------
    OBS_VECT_SIZE : int
        The size of the observation vector.

    Returns
    -------
    np.array
        The records numpy array.

    '''
    # first col for label, remaining OBS_VECT_SIZE cols for environment features
    return np.zeros((0, 5+obs_vect_size), dtype=np.float32)
    
if __name__ == '__main__':
    util.set_wd_to_package_root()
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_nothing_capacity_threshold",  help="The threshold " +
                        "max. line rho at which the tutor takes actions.",
                        required=False, default=.97,type=float)
    parser.add_argument("--disable_line",  help="The index of the line to be disabled.",
                        required=False,default=-1,type=int)
    parser.add_argument("--start_chronic_id",  help="The chronic to start with.",
                        required=False,default=1,type=int)
    args = parser.parse_args()
    
    config = util.load_config()
    #Load constants, settings, hyperparameters, argurments
    save_path = config['paths']['tutor_imitation']
    num_chronics = config['tutor_generated_data']['n_chronics']
    start_chronic_id = args.start_chronic_id
    do_nothing_capacity_threshold = args.do_nothing_capacity_threshold
    disable_line = args.disable_line
    
    #Initialize environment
    #NOTE: the first chronic, with id 0, is skipped as reset() is called after
    env = init_env(config)
    print("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))
    env.set_id(start_chronic_id)
    
    #Prepare tutor and record objects
    tutor = Tutor(env.action_space, get_env_actions(disable_line=disable_line),
                  args.do_nothing_capacity_threshold)
    obs_vect_size = len(env.get_obs().to_vect())
    records = empty_records(obs_vect_size)
    
    for num in range(start_chronic_id, start_chronic_id+num_chronics):
        
        #Reset variables
        obs = env.reset()
        done = False
        day_records = empty_records(obs_vect_size)
        
        #Disable lines, if any
        if disable_line != -1:
            obs, _, done, _ = env.step(env.action_space({"set_line_status":(disable_line,-1) }))
        
        print('current chronic: %s' % env.chronics_handler.get_name())
        reference_topo_vect = obs.topo_vect.copy()

        while not done:
            #reset topology at midnight, store days' records, reset days' records
            if obs.get_time_stamp().time()==dt.time(23,55):
                obs, _, done, _ = env.step(env.action_space({'set_bus': reference_topo_vect}))
                records = np.concatenate((records, day_records), axis=0)
                day_records = empty_records(obs_vect_size)
                continue
                
            #if not midnight, find a normal action
            action, idx, dn_rho, min_rho, time = tutor.act(obs)
            
            #don't store the action if the max. capacity is below the threshdold
            if idx != -2:
                # save a record
                day_records = np.concatenate((day_records, np.concatenate((
                                                [idx, dn_rho, min_rho, time, env.nb_time_step], 
                                                obs.to_vect())).astype(np.float32)[None, :]), axis=0)
                
            obs, _, done, _ = env.step(action)
         
        # print whether game was completed succesfully, save days' records if so
        if  env.nb_time_step == env.chronics_handler.max_timestep():
            print('game over (win) at step-%d\n\n\n' %  env.nb_time_step)
        else:
            print('game over (failure) at step-%d\n\n\n' %  env.nb_time_step)
            

        # save chronic records
        save_records(records,num,save_path,do_nothing_capacity_threshold,disable_line)
        records = empty_records(obs_vect_size)
        
    
          
# =============================================================================
#         # FOR TESTING PURPOSES ONLY
#         env.fast_forward_chronics(7488)
# =============================================================================
        
    

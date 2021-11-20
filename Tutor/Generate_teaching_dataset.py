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
    
    
def save_records(records: np.array, num: int, config: dict, lout: int = -1):
    '''
    Saves records of a chronic to disk and prints a message that they are saved. 

    Parameters
    ----------
    records : np.array
        The records.
    num : int
        INt representing the chronic which is saved.
    config : dict
        Dictionary with constant variables. Relevant here are the 
        do_nothing_capacity_threshold and the path where the file is to be saved.
    lout : int
        Index of any line that is out.
    '''
    dn_threshold = config['tutor_generated_data']['do_nothing_capacity_threshold']
    save_path = config['paths']['tutor_imitation']
    file_name = f'records_chronic:{num}_lout:{lout}' + \
                             f'_dnthreshold:{dn_threshold}.npy'
    np.save(os.path.join(save_path, file_name), records)
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
    return np.zeros((0, 1+obs_vect_size), dtype=np.float32)
    
if __name__ == '__main__':
    
    config = util.load_config()

    # parameters
    num_chronics = config['tutor_generated_data']['n_chronics']
    

    env = init_env(config)
    obs_vect_size = len(env.get_obs().to_vect())
    print("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))
    
    tutor = Tutor(env.action_space, get_env_actions())
    records = empty_records(obs_vect_size)
    
    for num in range(num_chronics):
        
        obs = env.reset()
        print('current chronic: %s' % env.chronics_handler.get_name())
        done, step = False, 0
        reference_topo_vect = obs.topo_vect.copy()
        day_records = empty_records(obs_vect_size)
        
        while not done:
            step += 1
            #reset topology at midnight, store days' records, reset days' records
            if obs.get_time_stamp().time()==dt.time(23,55):
                obs, _, done, _ = env.step(env.action_space({'set_bus': reference_topo_vect}))
                records = np.concatenate((records, day_records), axis=0)
                day_records = empty_records(obs_vect_size)
                continue
                
            #if not midnight, find a normal action
            action, idx = tutor.act(obs)
            
            #don't store the action if the max. capacity is below the threshdold
            if idx != -2:
                # save a record
                day_records = np.concatenate((day_records, np.concatenate(([idx], 
                                                obs.to_vect())).astype(np.float32)[None, :]), axis=0)
                
            obs, _, done, _ = env.step(action)
         
        # print whether game was completed succesfully, save days' records if so
        if step == env.chronics_handler.max_timestep():
            print('game over (win) at step-%d\n\n\n' % step)
        else:
            print('game over (failure) at step-%d\n\n\n' % step)
            

        # save chronic records
        save_records(records,num,config)
        records = empty_records(obs_vect_size)
        
    
          

    

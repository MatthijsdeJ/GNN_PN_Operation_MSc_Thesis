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
from imitation_generation.tutor import Tutor
from auxilary.generate_action_space import get_env_actions
import argparse
import auxilary.grid2op_util as g2o_util
import auxilary.util as util

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
    
def save_records(records: np.array, 
                 chronic: int, 
                 save_path: str, 
                 days_completed: int,
                 do_nothing_capacity_threshold: float, 
                 lout: int = -1,):
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
    days_completed : int
        The number of days completed.
    do_nothing_capacity_threshold : int
        The threshold max. line rho at which the tutor takes actions.
    lout : int
        Index of any line that is out.
    '''

    
    folder_name = f'records_chronics_lout:{lout}_dnthreshold:{do_nothing_capacity_threshold}'
    file_name = f'records_chronic:{chronic}_dayscomp:{days_completed}.npy'
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_nothing_capacity_threshold",  help="The threshold " +
                        "max. line rho at which the tutor takes actions.",
                        required=False, default=.97,type=float)
    parser.add_argument("--disable_line",  help="The index of the line to be disabled.",
                        required=False,default=-1,type=int)
    parser.add_argument("--start_chronic_id",  help="The chronic to start with.",
                        required=False,default=0,type=int)
    args = parser.parse_args()
    
    config = util.load_config()
    #Load constants, settings, hyperparameters, argurments
    save_path = config['paths']['tutor_imitation']
    num_chronics = config['tutor_generated_data']['n_chronics']
    ts_in_day = int(config['rte_case14_realistic']['ts_in_day'])
    start_chronic_id = args.start_chronic_id
    do_nothing_capacity_threshold = args.do_nothing_capacity_threshold
    disable_line = args.disable_line
    
    #Initialize environment
    env = g2o_util.init_env(config, grid2op.Rules.AlwaysLegal)
    print("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))
    env.set_id(start_chronic_id)
    
    #Prepare tutor and record objects
    tutor = Tutor(env.action_space, get_env_actions(disable_line=disable_line),
                  args.do_nothing_capacity_threshold)
    obs_vect_size = len(env.get_obs().to_vect())
    records = empty_records(obs_vect_size)
    
    #Auxilary ts_to_day function
    ts_to_day = lambda ts: g2o_util.ts_to_day(ts,ts_in_day)
    
    for num in range(start_chronic_id, start_chronic_id+num_chronics):
        
        #(Re)set variables
        obs = env.reset()
        done,days_completed = False, 0
        day_records = empty_records(obs_vect_size)
        
        #Disable lines, if any
        if disable_line != -1:
            obs, _, _, info = env.step(env.action_space(
                            {"set_line_status":(disable_line,-1) }))
        else:
            info = {'exception':[]}

        print('current chronic: %s' % env.chronics_handler.get_name())
        reference_topo_vect = obs.topo_vect.copy()

        while env.nb_time_step < env.chronics_handler.max_timestep():
            #Check for Diverging Powerflow exceptions, which happen sporadically
            if grid2op.Exceptions.PowerflowExceptions.DivergingPowerFlow in \
                        [type(e) for e in info['exception']]: 
                print(f'Powerflow exception at step {env.nb_time_step} '+
                      f'on day {ts_to_day(env.nb_time_step)}')
                info = g2o_util.skip_to_next_day(env, ts_in_day,
                                                 num, disable_line)
                day_records = empty_records(obs_vect_size)
                continue
                
            obs = env.get_obs()
            
            #reset topology at midnight, store days' records, reset days' records
            if env.nb_time_step%ts_in_day == ts_in_day-1:
                print(f'Day {ts_to_day(env.nb_time_step)} completed.')
                obs, _, _, _ = env.step(env.action_space({'set_bus': 
                                                        reference_topo_vect}))
                records = np.concatenate((records, day_records), axis=0)
                day_records = empty_records(obs_vect_size)
                days_completed += 1
                continue
                
            #if not midnight, find a normal action
            action, idx, dn_rho, min_rho, time = tutor.act(obs)
            
            #don't store the action if the max. capacity is below the threshdold
            if idx != -2:
                # save a record
                day_records = np.concatenate((day_records, np.concatenate((
                                [idx, dn_rho, min_rho, time, env.nb_time_step], 
                                obs.to_vect())).astype(np.float32)[None, :]), axis=0)
                
            obs, _, done, info = env.step(action)
            
            #If the game is done at this point, this indicated a (failed) game over
            #If so, reset the environment to the start of next day and discard the records
            if env.done:
                print(f'Failure at step {env.nb_time_step} '+
                      f'on day {ts_to_day(env.nb_time_step)}')
                info = g2o_util.skip_to_next_day(env,  ts_in_day, 
                                                 num, disable_line)
                day_records = empty_records(obs_vect_size)
         
        # print whether game was completed succesfully, save days' records if so
        print('Chronic exhausted! \n\n\n')
            
        
        # save chronic records
        save_records(records,num,save_path,days_completed,
                     do_nothing_capacity_threshold,disable_line)
        records = empty_records(obs_vect_size)
    
          
# =============================================================================
#         # FOR TESTING PURPOSES ONLY
#         env.fast_forward_chronics(7488)
# =============================================================================
        
    
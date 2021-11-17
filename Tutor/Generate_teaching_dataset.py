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
import time
import grid2op
import numpy as np
from Tutor import Tutor
import datetime as dt

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

def prepare_env(data_path: str, scenario_path: str, seed: int = 0) ->  \
                grid2op.Environment.Environment:
    '''
    Prepares the Grid2Op environment.

    Parameters
    ----------
    data_path : str
        String representing the path of the environment definition.
    scenario_path : str
        String representing the path of the scenario/chronic files.
    seed : int, optional
        Seed for generating random numers. The default is 0.

    Returns
    -------
    env : TYPE
        The Grid2Op environment.

    '''
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
        
    env.seed(seed)  # for reproducible experiments

    # thermal limits of Medha’s case
    thermal_limits = [1000,1000,1000,1000,1000,1000,1000, 760,450, 760,380,380,760,380,760,380,380,380,2000,2000]
    env.set_thermal_limit(thermal_limits)
    return env
    
    
def save_records(SAVE_PATH: str, records: np.array):
    '''
    Saves records to disk and prints a message that they are saved. 

    Parameters
    ----------
    SAVE_PATH : str
        String representation of the folder to save the records in.
    records : np.array
        The records.
    '''
    filepath = os.path.join(SAVE_PATH, 'records_%s.npy' % (time.strftime("%m-%d-%H-%M", time.localtime())))
    np.save(filepath, records)
    print('# records are saved! #')
    

def empty_records(OBS_VECT_SIZE: int):
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
    return np.zeros((0, 1+OBS_VECT_SIZE), dtype=np.float32)
    
if __name__ == '__main__':
    
    # environment definition
    DATA_PATH = '../Data/rte_case14_realistic'  # for demo only, use your own dataset
    SCENARIO_PATH = '../Data/rte_case14_realistic/chronics'
    SAVE_PATH = '../Data/tutor_generated_data'
    ACTION_SPACE_FILE = '../action_space/action_space.npy'
    # hyper-parameters
    NUM_CHRONICS = 1
    SAVE_CHRONIC_INTERVAL = 10
    OBS_VECT_SIZE = 437

    env = prepare_env(DATA_PATH, SCENARIO_PATH)
    print("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))
    
    tutor = Tutor(env.action_space, ACTION_SPACE_FILE)
    records = empty_records(OBS_VECT_SIZE)
    
    for num in range(NUM_CHRONICS):
        
        env.reset()
        print('current chronic: %s' % env.chronics_handler.get_name())
        done, step, obs = False, 0, env.get_obs()
        reference_topo_vect = obs.topo_vect.copy()
        day_records = empty_records(OBS_VECT_SIZE)
        
        while not done:
            step += 1
            #reset topology at midnight, store days' records, reset days' records
            if obs.get_time_stamp().time()==dt.time(23,55):
                obs, _, done, _ = env.step(env.action_space({'set_bus': reference_topo_vect}))
                records = np.concatenate((records, day_records), axis=0)
                day_records = empty_records(OBS_VECT_SIZE)
                continue
                
            #if not midnight, find a normal action
            action, idx = tutor.act(obs)
            
            #don't store the action if the max. capacity is below the threshdold
            if idx != -2:
                # save a record
                day_records = np.concatenate((day_records, np.concatenate(([idx], 
                                                obs.to_vect())).astype(np.float32)[None, :]), axis=0)
                
            obs, _, done, info = env.step(action)
         
        # print whether game was completed succesfully, save days' records if so
        if obs.rho.max()>=1:
            print('game over (failure) at step-%d\n\n\n' % step)
        else:
            print('game over (win) at step-%d\n\n\n' % step)
            records = np.concatenate((records, day_records), axis=0)

        # periodically save current records, reset records to avoid duplicates
        if (num + 1) % SAVE_CHRONIC_INTERVAL  == 0:
                save_records(SAVE_PATH, records)
                records = empty_records(OBS_VECT_SIZE)
        
    #save records when finished
    save_records(SAVE_PATH, records)
          

    

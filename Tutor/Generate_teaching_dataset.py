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
import ipdb

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

    
if __name__ == '__main__':
    
    # environment definition
    DATA_PATH = '../Data/rte_case14_realistic'  # for demo only, use your own dataset
    SCENARIO_PATH = '../Data/rte_case14_realistic/chronics'
    SAVE_PATH = '../Data/tutor_generated_data'
    ACTION_SPACE_FILE = '../action_space/action_space.npy'
    # hyper-parameters
    NUM_CHRONICS = 1
    SAVE_INTERVAL = 10
    OBS_VECT_SIZE = 437

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
        
    print(env.chronics_handler.get_id()) # get folder where current scenario is located
    print("Number of available scenarios " + str(len(env.chronics_handler.subpaths)))
    
    env.seed(0)  # for reproducible experiments

    # thermal limits of Medha’s case
    thermal_limits = [1000,1000,1000,1000,1000,1000,1000, 760,450, 760,380,380,760,380,760,380,380,380,2000,2000]
    env.set_thermal_limit(thermal_limits)

    tutor = Tutor(env.action_space, ACTION_SPACE_FILE)
    # first col for label, remaining OBS_VECT_SIZE cols for feature (observation.to_vect())
    records = np.zeros((1, 1+OBS_VECT_SIZE), dtype=np.float32)
    for num in range(NUM_CHRONICS):
        
        env.reset()
        print('current chronic: %s' % env.chronics_handler.get_name())
        done, step, obs = False, 0, env.get_obs()
        reference_topo_vect = obs.topo_vect.copy()
        
        while not done:
            #reset topology at midnight
            if obs.get_time_stamp().time()==dt.time(23,55):
                obs, _, done, _ = env.step(env.action_space({'set_bus': reference_topo_vect}))
                continue
                
            #if not midnight, take normal actione
            action, idx = tutor.act(obs)
            #don't store the action if the max. capacity is below the threshdold
            if idx != -2:
                # save a record
                records = np.concatenate((records, np.concatenate(([idx], obs.to_vect())).astype(np.float32)[None, :]), axis=0)
            obs, _, done, _ = env.step(action)
            step += 1
            
        print('game over at step-%d\n\n\n' % step)

        # save current records
        if (num + 1) % SAVE_INTERVAL == 0:
            filepath = os.path.join(SAVE_PATH, 'records_%s.npy' % (time.strftime("%m-%d-%H-%M", time.localtime())))
            np.save(filepath, records)
            print('# records are saved! #')
          
        filepath = os.path.join(SAVE_PATH, 'records_%s.npy' % (time.strftime("%m-%d-%H-%M", time.localtime())))
        np.save(filepath, records)
        print('# DONE; records are saved! #')
    

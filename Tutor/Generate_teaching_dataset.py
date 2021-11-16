"""
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


if __name__ == '__main__':
    # environment definition
    DATA_PATH = '../data/rte_case14_realistic'
    SCENARIO_PATH = '../data/rte_case14_realistic/chronics'
    SAVE_PATH = '../training_data'
    ACTION_SPACE_DIRECTORY = '../action_space'
    # hyper-parameters
    NUM_CHRONICS = 100
    SAVE_INTERVAL = 10

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend, test=True)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, test=True)
        
    print(env.chronics_handler.get_id()) # get folder where current scenario is located
    print("Number of available scenarios " + str(len(env.chronics_handler.subpaths)))
    
    env.seed(0)  # for reproducible experiments

    # thermal limits of Medha’s case
    thermal_limits = [1000,1000,1000,1000,1000,1000,1000, 760,450, 760,380,380,760,380,760,380,380,380,2000,2000]
    env.set_thermal_limit(thermal_limits)

    tutor = Tutor(env.action_space, ACTION_SPACE_DIRECTORY)
    # first col for label, remaining 1266 cols for feature (observation.to_vect())
    records = np.zeros((1, 1267), dtype=np.float32)
    for num in range(NUM_CHRONICS):
        env.reset()
        print('current chronic: %s' % env.chronics_handler.get_name())
        done, step, obs = False, 0, env.get_obs()
        while not done:
            action, idx = tutor.act(obs)
            if idx != -1:
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

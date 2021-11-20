"""
ADAPTED FROM:
https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution/blob/master/Tutor/Tutor.py

Protected by Mozilla Public License Version 2.0.

In this file, an expert agent (named Tutor), which does a greedy search
in the reduced action space is built.
It receives an observation, and returns the action that decreases the rho
most, as well as its index [api: Tutor.act(obs)].

original author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import numpy as np
import grid2op
from grid2op.Agent import BaseAgent
from typing import Tuple
import util


class Tutor(BaseAgent):
    def __init__(self, env_action_space, selected_action_space):
        BaseAgent.__init__(self, action_space=env_action_space)
        self.actions = selected_action_space
        
        config = util.load_config()
        self.do_nothing_capacity_threshold = config['tutor_generated_data']['do_nothing_capacity_threshold']

# =============================================================================
#     @staticmethod
#     def reconnect_array(obs):
#         new_line_status_array = np.zeros_like(obs.rho)
#         disconnected_lines = np.where(obs.line_status==False)[0]
#         for line in disconnected_lines[::-1]:
#             if not obs.time_before_cooldown_line[line]:
#                 # this line is disconnected, and, it is not cooling down.
#                 line_to_reconnect = line
#                 new_line_status_array[line_to_reconnect] = 1
#                 break  # reconnect the first one
#         return new_line_status_array
# =============================================================================

# =============================================================================
#     def array2action(self, array: np.array) -> grid2op.Action.TopologyAction:
#         '''
#         Turns an array representing a set action into the corresponding
#         topology-action.
# 
#         Parameters
#         ----------
#         array: np.array
#             The array representing the set action.
# 
#         Returns
#         -------
#         action : grid2op.Action.TopologyAction
#             The topology action.
#         '''
#         action = self.action_space({'set_bus': array})
#         return action
# 
# =============================================================================

# =============================================================================
#     @staticmethod
#     def is_legal(action, obs):
#         substation_to_operate = int(action.as_dict()['change_bus_vect']['modif_subs_id'][0])
#         if obs.time_before_cooldown_sub[substation_to_operate]:
#             # substation is cooling down
#             return False
#         for line in [eval(key) for key, val in action.as_dict()['change_bus_vect'][str(substation_to_operate)].items() if 'line' in val['type']]:
#             if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
#                 # line is cooling down, or line is disconnected
#                 return False
#         return True
# =============================================================================

    def act(self, observation: grid2op.Observation.CompleteObservation) \
                        -> Tuple[grid2op.Action.TopologyAction,int]:
        '''
        For a particular observation, searches through the action space with 
        a greedy strategy to find the action that minimizes the max. 
        (over the power lines) capacity in simulation.

        Parameters
        ----------
        observation : grid2op.Observation.CompleteObservation
            The observation.

        Returns
        -------
        Tuple(grid2op.Action.TopologyAction,int)
            The selected set action and the index of the selected action.
            In case that the max. line capacity is below self.do_nothing_capacity_threshold,
            no action is selected, and the the index is -2.
            In case that the max. line capacity is above self.do_nothing_capacity_threshold,
            but no set action is selected, the selected action is do-nothing
            and the index is -1.
        '''
        tick = time.time()

        if observation.rho.max() < self.do_nothing_capacity_threshold:
            # secure, return "do nothing" in bus switches.
            return self.action_space(), -2

        #not secure, do a greedy search
        print('%s: close to overload! line-%d has a max. rho of %.2f' % 
              (str(observation.get_time_stamp()), observation.rho.argmax(), observation.rho.max()))
        
        #default action is do nothing
        action_chosen = self.action_space()
        obs, _, done, _ = observation.simulate(action_chosen)
        min_rho = obs.rho.max()
        return_idx = -1

        for idx, a in enumerate(self.actions):
            
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                return_idx = idx
    
        print('Action %d results in a forecasted max. rho of %.2f, search duration is %.2fs' 
              % (return_idx, min_rho, time.time() - tick))
        return action_chosen, return_idx

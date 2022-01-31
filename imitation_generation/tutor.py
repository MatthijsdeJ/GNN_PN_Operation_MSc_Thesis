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
import time
import grid2op
from grid2op.Agent import BaseAgent
from typing import Tuple, Optional, Sequence
from abc import ABC, abstractmethod
from statistics import mean
import numpy as np


class Strategy(ABC):
    """
    Base class for the strategy taken by the tutor model.
    """

    @abstractmethod
    def select_act(self,
                   action_space: Sequence[grid2op.Action.TopologyAction],
                   observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, int, float]:
        """
        Selects an action.

        Parameters
        ----------
        action_space : Sequence[grid2op.Action.TopologyAction]
            The available actions.
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        action_idx : int
            The index of the selected action. -1 is the do-nothing action.
        sel_rho : float
            The rho value resulting from the selected action.
        """
        pass

    @staticmethod
    def is_do_nothing_set_bus(topo_vect: np.array, set_bus: np.array) -> bool:
        """
        Checks if a set_bus act results in the same topo vect (i.e. is a do-nothing action).

        Parameters
        ----------
        topo_vect : np.array
            Array representing the current configuration of objects to bus-bars.
        set_bus : np.array
            Array representing the set_bus action.

        Returns
        -------
        bool
            Whether the set_bus actions results in the same topo vect.
        """
        return all(np.logical_or(set_bus == 0, set_bus == topo_vect))


class GreedyStrategy(Strategy):
    """
    Greedy strategy that always selects the action that minimizes the max. rho in the simulated next timestep.
    """

    def __init__(self, do_nothing_action: grid2op.Action.BaseAction):
        """
        Parameters
        ----------
        do_nothing_action : grid2op.Action.BaseAction
            The do-nothing action.
        """
        super()
        self.do_nothing_action = do_nothing_action

    def select_act(self,
                   action_space: Sequence[grid2op.Action.TopologyAction],
                   observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, int, float]:
        """
        Selects an action based on the greedy strategy: the action that minimizes the max. rho in the simulated
        next timestep is selected.

        Parameters
        ----------
        action_space : Sequence[grid2op.Action.TopologyAction]
            The available actions.
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        action_idx : int
            The index of the selected action. -1 is the do-nothing action.
        sel_rho : float
            The rho value resulting from the selected action.
        """
        # default action is do nothing
        action_chosen = self.do_nothing_action
        obs, _, _, _ = observation.simulate(action_chosen)
        sel_rho = obs.rho.max()
        action_idx = -1
        # TODO: there is no check to ensure that the default action does not lead to a game-over. Since a game-over
        #   leads to max. rhos of zero, which is always the minimum, any actions that would not lead to a game-over
        #   would not be selected.

        # simulate each action
        for idx, a in enumerate(action_space):
            obs, _, done, _ = observation.simulate(a)

            # if an action leads to a game-over, skip it
            if done:
                continue

            # if an action results in the lowest max. rho so far, store it as the best action so far
            if obs.rho.max() < sel_rho:
                sel_rho = obs.rho.max()
                action_chosen = a
                action_idx = idx

        return action_chosen, action_idx, sel_rho


class CheckNMinOneStrategy(Strategy):
    """
    Strategy that selects an action based on the robustness of that action's resulting topology to line outages.
    """

    def __init__(self,
                 env_action_space: grid2op.Action.ActionSpace,
                 line_outages_to_consider: Sequence[int],
                 N0_max_rho: float = 1.0):
        """
        Parameters
        ----------
        env_action_space: grid2op.Action.ActionSpace
            The full action space of the environment. Can be used to disable lines for N-1 scenarios, or to generate
            an empty action.
        line_outages_to_consider : Sequence[int]
            The indices of the lines whose outages to check with.
        N0_max_rho : float
            Actions that lead to a bad N0 scenario (i.e. a scenario where no line outages are considered) are
            not further evaluated. This value determines the max. rho threshold: if an action causes the max. rho in
            the N0 scenario to exceed it, that action is not further evaluated and not selected.
        """
        super()
        self.env_action_space = env_action_space
        self.line_outages_to_consider = line_outages_to_consider
        self.N0_max_rho = N0_max_rho

    def mean_max_rho_over_NMinOne(self,
                                  a: grid2op.Action.BaseAction,
                                  observation: grid2op.Observation.CompleteObservation) -> float:
        """
        Given an action, calculate the mean (over multiple N-1 scenarios) of the max. rho (over the power lines)
        of the observations produced by simulating that action.

        Parameters
        ----------
        a : grid2op.Action.BaseAction
            The action to calculate the mean for.
        observation
            The current observation, on which to simulate the action.

        Returns
        -------
        float
            The mean over the max. rhos, as described above.
        """
        set_bus = a.set_bus

        max_rhos = []
        # Iterate over N-1 scenarios
        for line_idx in self.line_outages_to_consider:
            # To consider the N-1 scenario, we include disabling a line as part of the action
            combined_action = self.env_action_space({"set_line_status": (line_idx, -1),
                                                     "set_bus": set_bus})

            # Simulate the action to obtain the max. rho, add it to the list
            obs, _, done, _ = observation.simulate(combined_action)
            max_rhos.append(obs.rho.max())
            # TODO: what if one of the line outages causes a game-over or diverging powerflow?

        return mean(max_rhos)

    def select_act(self,
                   action_space: Sequence[grid2op.Action.TopologyAction],
                   observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, int, float]:
        """
        Selects an action based on the performance of the action under N-1 scenarios. The action is selected based on:
            1) it minimizes the mean max. rho over the different N-1 scenarios obtained by disabling the lines in
            line_outage_to_consider.
            2) given that in the N0 scenario, the max. rho does not exceed N0_max_rho.

        Parameters
        ----------
        action_space : Sequence[grid2op.Action.TopologyAction]
            The available actions.
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        action_idx : int
            The index of the selected action. -1 is the do-nothing action.
        sel_rho : float
            The rho value resulting from the selected action.
        """
        # default action is do nothing
        action_chosen = self.env_action_space()
        obs, _, done, _ = observation.simulate(action_chosen)
        sel_rho = obs.rho.max()
        min_mean_max_rho_over_NMinOne = self.mean_max_rho_over_NMinOne(action_chosen,
                                                                       observation)
        action_idx = -1
        # TODO: there is no check to ensure that the default action does not lead to a game-over or a diverging power-
        #  flow. Since a game-over leads to max. rho of zero, which is always the minimum, any actions that would not
        #  lead to a game-over would not be selected.

        # simulate each action
        for idx, a in enumerate(action_space):
            # If the action is a do-nothing action, skip it
            if self.is_do_nothing_set_bus(observation.topo_vect, a.set_bus):
                continue

            # Simulate the action
            obs, _, done, _ = observation.simulate(a)

            # if an action leads to a game-over or causes max. rho to exceed the N0 max. rho threshold, skip it
            if done or obs.rho.max() > self.N0_max_rho:
                continue

            # if an action decreases the mean (over the N-1 scenarios) max. (over the power lines) further than any
            # action so far, store the action and its information
            mean_max_rho_over_NMinOne = self.mean_max_rho_over_NMinOne(a, observation)
            if mean_max_rho_over_NMinOne < min_mean_max_rho_over_NMinOne:
                min_mean_max_rho_over_NMinOne = mean_max_rho_over_NMinOne
                action_chosen = a
                action_idx = idx
                sel_rho = obs.rho.max()

        return action_chosen, action_idx, sel_rho


class Tutor(BaseAgent):
    def __init__(self,
                 env_action_space: grid2op.Action.ActionSpace,
                 selected_action_space: Sequence[grid2op.Action.TopologyAction],
                 do_nothing_capacity_threshold: float,
                 strategy: Strategy):
        """
        Parameters
        ----------
        env_action_space : grid2op.Action.ActionSpace
            The full action space of the environment.
        selected_action_space : Sequence[grid2op.Action.TopologyAction]
            The selected actions for the agent to try.
        do_nothing_capacity_threshold : float
            The rho value, so that if not exceeded by any line a do-nothing action is selected.
        strategy : Strategy
            The strategy to use for selecting actions.
        """
        BaseAgent.__init__(self, action_space=env_action_space)
        self.actions = selected_action_space
        self.do_nothing_capacity_threshold = do_nothing_capacity_threshold
        self.strategy = strategy

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
    #         for line in [eval(key) for key, val in action.as_dict()['change_bus_vect']
    #         [str(substation_to_operate)].items()
    #         if 'line' in val['type']]:
    #             if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
    #                 # line is cooling down, or line is disconnected
    #                 return False
    #         return True
    # ====================s=========================================================

    def act(self, observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, int, Optional[float],
                     Optional[float], Optional[float]]:
        """
        For a particular observation, searches through the action space with 
        a greedy strategy to find the action that minimizes the max. 
        (over the power lines) capacity in simulation.

        Parameters
        ----------
        observation : grid2op.Observation.CompleteObservation
            The observation.

        Returns
        -------
        action : grid2op.Action.BaseAction
            The selected action.
        action_idx : int
            The index of the selected action.
            In case that the max. line capacity is below self.do_nothing_capacity_threshold,
            no action is selected, and the index is -2.
            In case that the max. line capacity is above self.do_nothing_capacity_threshold,
            but no set action is selected, the selected action is do-nothing
            and the index is -1.
        dn_rho : float
            The rho obtained from simulating a do-nothing action.
            None if the max. line capacity is below self.do_nothing_capacity_threshold.
        sel_rho : float
            The rho obtained by the selected action.
            None if the max. line capacity is below self.do_nothing_capacity_threshold.
        time : float
            The elapsed time of the function in seconds.
            None if the max. line capacity is below self.do_nothing_capacity_threshold.
        """
        tick = time.time()

        if observation.rho.max() < self.do_nothing_capacity_threshold:
            # secure, return "do nothing" in bus switches.
            return self.action_space(), -2, None, None, None

        # not secure, do a greedy search
        print('%s: close to overload! line-%d has a max. rho of %.2f' %
              (str(observation.get_time_stamp()), observation.rho.argmax(), observation.rho.max()))

        # calculate the max. rho of the do-nothing action
        do_nothing_action = self.action_space()
        obs, _, _, _ = observation.simulate(do_nothing_action)
        dn_rho = obs.rho.max()

        # select an action based on the strategy
        action_chosen, action_idx, sel_rho = self.strategy.select_act(self.actions,
                                                                      observation)

        # print the selected action, return the results
        print('Action %d results in a forecasted max. rho of %.2f, search duration is %.2fs'
              % (action_idx, sel_rho, time.time() - tick))
        return action_chosen, action_idx, dn_rho, sel_rho, time.time() - tick

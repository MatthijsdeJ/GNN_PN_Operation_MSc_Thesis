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

    @staticmethod
    def get_max_rho_simulated(observation: grid2op.Observation.CompleteObservation,
                              action: grid2op.Action.BaseAction) -> float:
        """
        Simulates an action, and gets the max. rho. Returns infinity in case of a game-over.

        Parameters
        ----------
        observation: grid2op.Observation.CompleteObservation
            The observation to simulate the action in.
        action: grid2op.Action.BaseAction
            The action to simulate.

        Returns
        -------
        float
            The max. rho of the observation resulting from the simulation of the action. Infinity in case of a
            game-over.
        """
        obs, _, done, _ = observation.simulate(action)
        return obs.rho.max() if not done else float('Inf')
    
    @staticmethod
    def activity_criterion_current(observation: grid2op.Observation.CompleteObservation,
                              do_nothing_capacity_threshold: float) -> bool:
        """
        Evaluates activity criterion 1. Returns if agent should get active.
        
        Parameters
        ----------
        observation: grid2op.Observation.CompleteObservation
            The observation to simulate the action in.
        do_nothing_capacity_threshold: float
            The rho value, so that if exceeded by any line the agent gets active.

        Returns
        -------
        bool
            Whether the agent should get active.
        """
        return observation.rho.max() > do_nothing_capacity_threshold

    @staticmethod
    def activity_criterion_simulate(observation: grid2op.Observation.CompleteObservation,
                             do_nothing_action: grid2op.Action.BaseAction
                             do_nothing_capacity_threshold: float) -> bool:
        """
        Simulates the do-nothing action, evaluates activity criterion 2. Returns if agent should get active.
        
        Parameters
        ----------
        observation: grid2op.Observation.CompleteObservation
            The observation to simulate the action in.
        do_nothing_action: grid2op.Action.BaseAction
            The do nothing action that will be simulated.
        do_nothing_capacity_threshold: float
            The rho value, so that if exceeded by any line the agent gets active.

        Returns
        -------
        bool
            Whether the agent should get active.
        """
        obs, _, done, _ = observation.simulate(do_nothing_action)
        return obs.rho.max() > self.do_nothing_capacity_threshold or done

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
        # The default action is do-nothing
        action_chosen = self.do_nothing_action
        selected_rho = self.get_max_rho_simulated(observation, action_chosen)
        action_idx = -1

        # Simulate each action
        # TODO: This could be sped up by skipping implicit do nothing actions
        for idx, a in enumerate(action_space):
            # Obtain the max. rho of the observation resulting from the simulated action
            action_rho = self.get_max_rho_simulated(observation, a)

            # If an action results in the lowest max. rho so far, store it as the best action so far
            if action_rho < selected_rho:
                selected_rho = action_rho
                action_chosen = a
                action_idx = idx

        return action_chosen, action_idx, selected_rho


class CheckNMinOneStrategy(Strategy):
    """
    Strategy that selects an action based on the robustness of that action's resulting topology to line outages.
    """

    def __init__(self,
                 env_action_space: grid2op.Action.ActionSpace,
                 line_outages_to_consider: Sequence[int],
                 N0_rho_threshold: float = 1.0):
        """
        Parameters
        ----------
        env_action_space: grid2op.Action.ActionSpace
            The full action space of the environment. Can be used to disable lines for N-1 scenarios, or to generate
            an empty action.
        line_outages_to_consider : Sequence[int]
            The indices of the lines whose outages to check with.
        N0_rho_threshold : float
            Actions that lead to a good N0 scenario (i.e. a scenario where no line outages are considered) are
            first evaluated on their N-1 performance. This value determines what counts as 'good' N0 performance:
            if an action has a N0 max. rho that exceeds it, that actions' N-1 max. max. rho is not evaluated.
        """
        super()
        self.env_action_space = env_action_space
        self.line_outages_to_consider = line_outages_to_consider
        self.N0_rho_threshold = N0_rho_threshold

    def max_max_rho_NMinOne(self,
                            a: grid2op.Action.BaseAction,
                            observation: grid2op.Observation.CompleteObservation) -> float:
        """
        Given an action, calculate the max. (over multiple N-1 scenarios) of the max. rho (over the power lines)
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
            The max over the max. rhos, as described above.
        """
        set_bus = a.set_bus

        max_rhos = []
        # Iterate over N-1 scenarios
        for line_idx in self.line_outages_to_consider:
            # To consider the N-1 scenario, we include disabling a line as part of the action
            combined_action = self.env_action_space({"set_line_status": (line_idx, -1),
                                                     "set_bus": set_bus})

            # Simulate the action to obtain the max. rho, add it to the list
            max_rhos.append(self.get_max_rho_simulated(observation, combined_action))

        return max(max_rhos)

    def select_act(self,
                   action_space: Sequence[grid2op.Action.TopologyAction],
                   observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, int, float]:
        """
        Selects an action based on the performance of the action under N-1 scenarios. The action is selected based on:
            1) if there are any actions that result in a N0 max. rho under the N0 rho threshold, select the among the
            actions satisfying that condition, the one with the lowest N-1 max. max. rho threshold, provided that
            this is not infinity.
            2) if no action has a N0 max. rho under the N0 threshold, OR all actions satisfying that condition
            have a N-1 max. max. rho threshold of infinity, then select the action that minimizes the N0 max. rho
            threshold.

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
        action_chosen = sel_rho = action_idx = None

        # Select the do-something actions
        actions = [(idx, a) for idx, a in enumerate(action_space)
                   if not self.is_do_nothing_set_bus(observation.topo_vect, a.set_bus)]
        # Add back singular do-nothing action at the start
        actions.insert(0, (-1, self.env_action_space()))
        # Calculate N-0 max. rho per action
        action_max_rho_tuples = [(idx, a, self.get_max_rho_simulated(observation, a)) for idx, a in actions]
        # Select the actions with a N-0 max. rho below the N-0. max. rho threshold
        action_max_rho_tuples_below_threshold = [(idx, a, max_rho) for idx, a, max_rho in action_max_rho_tuples
                                                 if max_rho < self.N0_rho_threshold]

        # If there are actions with a N-0 rho below the N-0 rho threshold,
        # select the one among them with the best N-1 max. max. rho
        # provided that this is not infinity
        if action_max_rho_tuples_below_threshold:
            lowest_max_max_rho_NMinOne = float('inf')

            for idx, a, max_rho in action_max_rho_tuples_below_threshold:
                # Calculate the N-1 max. max. rho
                max_max_rho_NMinOne = self.max_max_rho_NMinOne(a, observation)

                # Set action as best action if it has the lowest N-1 max. max. rho so far
                if lowest_max_max_rho_NMinOne > max_max_rho_NMinOne:
                    action_chosen = a
                    action_idx = idx
                    sel_rho = max_rho
                    lowest_max_max_rho_NMinOne = max_max_rho_NMinOne

            assert lowest_max_max_rho_NMinOne == float('inf') if action_chosen is None else True, \
                   "If no action is selected, then the lowest N-1 max. max. rho must be infinity."

        # At this point, either no action is selected or this action has a N0 max. rho below the N0 threshold.
        assert action_chosen is None or sel_rho < self.N0_rho_threshold, "At this point, action chosen should be" \
                                                                         "None or the the sel_rho below the threshold."

        # If the best action so far still has one scenario that fails in the N-1 max. max. rho calculation,
        # i.e. if the best N-1 max. max. rho is still infinite, then select the action with the best N-0 max. rho
        if action_chosen is None:
            assert sel_rho is None and action_idx is None, "Action chosen is none, but other variables not."

            # Select action with best N-0 max. rho
            for idx, a, max_rho in action_max_rho_tuples:

                # Set the do-nothing action as the default: this works because the first entry in the action list
                # is the do-nothing action
                if action_chosen is None:
                    assert idx == -1, "First action should be the do-nothing action."
                    action_idx, action_chosen, sel_rho = idx, a, max_rho

                # If the N0 max. rho is lower than that for any action before, set the action to the current
                if sel_rho > max_rho:
                    action_idx, action_chosen, sel_rho = idx, a, max_rho

        # Assert postconditions
        assert not (action_chosen is None or sel_rho is None or action_idx is None), "One of the output variables is" \
                                                                                     "None."
        assert sel_rho >= 0, "Sel_rho cannot be negative"
        assert len(action_space) > action_idx >= -1, "Action idx is outside of it's possible range."
        assert action_idx == -1 if sel_rho == np.float("inf") else True,\
               "If sel_rho is infinite, the action should be do_nothing."

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
        For a particular observation, select an action according to its strategy.

        Parameters
        ----------
        observation : grid2op.Observation.CompleteObservation
            The observation.

        Returns
        -------
        selected_action : grid2op.Action.BaseAction
            The selected action.
        selected_action_idx : int
            The index of the selected action.
            In case that the max. line capacity is below self.do_nothing_capacity_threshold,
            no action is selected, and the index is -2.
            In case that the max. line capacity is above self.do_nothing_capacity_threshold,
            but no set action is selected, the selected action is do-nothing
            and the index is -1.
        do-nothing rho : float
            The rho obtained from simulating a do-nothing action.
            None if the max. line capacity is below self.do_nothing_capacity_threshold.
        selected_rho : float
            The rho obtained by the selected action.
            None if the max. line capacity is below self.do_nothing_capacity_threshold.
        time : float
            The elapsed time of the function in seconds.
            None if the max. line capacity is below self.do_nothing_capacity_threshold.
        """
        tick = time.time()

        # Do nothing if the max. rho is below the max. rho threshold
        if observation.rho.max() < self.do_nothing_capacity_threshold:
            return self.action_space(), -2, None, None, None

        # If above that max. rho threshold, display a message
        print('%s: close to overload! line-%d has a max. rho of %.2f' %
              (str(observation.get_time_stamp()), observation.rho.argmax(), observation.rho.max()))

        # Calculate the max. rho of the do-nothing action
        do_nothing_action = self.action_space()
        obs, _, _, _ = observation.simulate(do_nothing_action)
        do_nothing_rho = obs.rho.max()

        # Select an action based on the strategy
        selected_action, selected_action_idx, selected_rho = self.strategy.select_act(self.actions, observation)

        # Print the selected action, return the results
        print('Action %d results in a forecasted max. rho of %.2f, search duration is %.2fs'
              % (selected_action_idx, selected_rho, time.time() - tick))
        return selected_action, selected_action_idx, do_nothing_rho, selected_rho, time.time() - tick

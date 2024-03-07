from abc import ABC, abstractmethod
import grid2op
import torch
import auxiliary.grid2op_util as g2o_util
import numpy as np
from auxiliary.generate_action_space import get_env_actions

class AgentStrategy(ABC):
    """
    Base class for the strategy used for evaluation.
    """

    # @property
    # @abstractmethod
    # def model(self):
    #     """
    #     Require declaration of the model attribute.
    #     """
    #     pass

    @abstractmethod
    def select_action(self, observation: grid2op.Observation.CompleteObservation) -> grid2op.Action.BaseAction:
        """
        Selects an action.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        """
        pass

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


class IdleStrategy(AgentStrategy):
    """
    Strategy that produces only do-nothing actions.
    """

    def __init__(self, action_space):
        super()
        self.action_space = action_space

    def select_action(self, observation: grid2op.Observation.CompleteObservation):
        """
        Selects an action.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        """
        action = self.action_space({})

        return action


class NaiveStrategy(AgentStrategy):
    """
    Naive strategy that simply selects the action predicted by the ML model.
    """

    def __init__(self, model, feature_statistics, action_space, dn_threshold: float):
        super()
        self.model = model
        self.feature_statistics = feature_statistics
        self.action_space = action_space
        self.dn_threshold = dn_threshold

    def select_action(self, observation: grid2op.Observation.CompleteObservation):
        """
        Selects an action.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        """
        if observation.rho.max() > self.dn_threshold:
            P = torch.flatten(self.model.predict_observation(observation, self.feature_statistics)).detach()
            P_sub_mask, _ = g2o_util.select_single_substation_from_topovect(P,
                                                                            observation.sub_info,
                                                                            f=lambda x: torch.sum(torch.clamp(x - 0.5,
                                                                                                              min=0)))
            P = np.array([1 if (m and p > 0.5) else 0 for m, p in zip(P_sub_mask, P)])
            set_action = 1 + (observation.topo_vect - 1 + P) % 2
            # TODO: Fix with disabled lines

            action = self.action_space({'set_bus': set_action})
        else:
            action = self.action_space({})

        return action


class VerifyStrategy(AgentStrategy):
    """
    Strategy that selects the action predicted by the ML model, but might reject it the action increases the max. rho
    over a threshold.
    """

    def __init__(self, model, feature_statistics, action_space, dn_threshold: float, reject_action_threshold: float):
        super()
        self.model = model
        self.feature_statistics = feature_statistics
        self.action_space = action_space
        self.dn_threshold = dn_threshold
        self.reject_action_threshold = reject_action_threshold

    def select_action(self, observation: grid2op.Observation.CompleteObservation):
        """
        Selects an action.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        """
        if observation.rho.max() > self.dn_threshold:
            P = torch.flatten(self.model.predict_observation(observation, self.feature_statistics)).detach()
            P_sub_mask, _ = g2o_util.select_single_substation_from_topovect(P,
                                                                            observation.sub_info,
                                                                            f=lambda x: torch.sum(torch.clamp(x - 0.5,
                                                                                                              min=0)))
            P = np.array([1 if (m and p > 0.5) else 0 for m, p in zip(P_sub_mask, P)])
            set_action = 1 + (observation.topo_vect - 1 + P) % 2
            # TODO: Test with disabled lines

            action = self.action_space({'set_bus': set_action})

            # Verify the action; if failed, use a do_nothing action
            simulation_max_rho = self.get_max_rho_simulated(observation, action)
            if simulation_max_rho > self.reject_action_threshold and simulation_max_rho > observation.rho.max():
                action = self.action_space({})
        else:
            action = self.action_space({})

        return action


class HybridStrategy(AgentStrategy):
    """
    Strategy that selects the action selected by the ML model, with two exceptions:
    1) Actions are rejected if they increase the max. rho over a threshold.
    2) Above a certain threshold, the agent takes actions using greedy simulation.
    """

    def __init__(self,
                 model,
                 feature_statistics,
                 action_space,
                 dn_threshold: float,
                 reject_action_threshold: float,
                 greedy_control_threshold: float,
                 greedy_action_space):
        super()
        self.model = model
        self.feature_statistics = feature_statistics
        self.action_space = action_space
        self.dn_threshold = dn_threshold
        self.reject_action_threshold = reject_action_threshold
        self.greedy_control_threshold = greedy_control_threshold
        self.greedy_action_space = greedy_action_space

    def select_action(self, observation: grid2op.Observation.CompleteObservation):
        """
        Selects an action.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        """
        if observation.rho.max() > self.dn_threshold:
            if observation.rho.max() < self.greedy_control_threshold:
                P = torch.flatten(self.model.predict_observation(observation, self.feature_statistics)).detach()
                P_sub_mask, _ = g2o_util.select_single_substation_from_topovect(P,
                                                                                observation.sub_info,
                                                                                f=lambda x:
                                                                                torch.sum(torch.clamp(x - 0.5, min=0)))
                P = np.array([1 if (m and p > 0.5) else 0 for m, p in zip(P_sub_mask, P)])
                set_action = 1 + (observation.topo_vect - 1 + P) % 2
                # TODO: Test with disabled lines

                action = self.action_space({'set_bus': set_action})

                # Verify the action; if failed, use a do_nothing action
                simulation_max_rho = self.get_max_rho_simulated(observation, action)
                if simulation_max_rho > self.reject_action_threshold and simulation_max_rho > observation.rho.max():
                    action = self.action_space({})
            else:
                # Greedy agent takes the wheel
                best_action = self.action_space({})
                best_rho = self.get_max_rho_simulated(observation, best_action)

                # Simulate each action
                for action in self.greedy_action_space:
                    # Obtain the max. rho of the observation resulting from the simulated action
                    action_rho = self.get_max_rho_simulated(observation, action)

                    # If an action results in the lowest max. rho so far, store it as the best action so far
                    if action_rho < best_rho:
                        best_rho = action_rho
                        best_action = action

                action = best_action
        else:
            action = self.action_space({})

        return action

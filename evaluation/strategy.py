from abc import ABC, abstractmethod
import grid2op
import torch
import auxiliary.grid2op_util as g2o_util
import numpy as np


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

        Returns
        -------

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

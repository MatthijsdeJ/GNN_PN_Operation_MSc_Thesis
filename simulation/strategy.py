from abc import ABC, abstractmethod
import grid2op
import torch
import auxiliary.grid2op_util as g2o_util
import numpy as np
from typing import Tuple, Optional, Sequence

import warnings

from auxiliary.config import get_config, NetworkType
import training.models
from training import postprocessing
from training.models import Model, FCNN, GCN
from data_preprocessing_analysis.data_preprocessing import reduced_env_variables, extract_gen_features, \
    extract_load_features, extract_or_features, extract_ex_features


class AgentStrategy(ABC):
    """
    Base class for the strategy used for simulation.
    """

    # @property
    # @abstractmethod
    # def model(self):
    #     """
    #     Require declaration of the model attribute.
    #     """
    #     pass

    @abstractmethod
    def select_action(self,
                      observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, Optional[dict]]:
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
        datapoint_dict : Optional[dict]
            A dictionary which contains information about the action. Used to save imitation learning tutor datapoints.
            The format of the dictionary is described in create_datapoint_dict(). It is the responsibility of the
            strategy to determine which actions are saved as datapoints; actions that should not be saved should return
            None.
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

    @staticmethod
    def create_datapoint_dict(action_index: int,
                              obs: grid2op.Observation.CompleteObservation,
                              do_nothing_rho: float,
                              action_rho: float,
                              duration: int = 0):
        """
        Create the dictionary for the datapoint.

        Parameters
        ----------
        action_index : int
            The index of the selected action in the auxiliary.generate_action_space.get_env_actions list.
        obs : grid2op.Observation.CompleteObservation
            The observation in which the action was selected.
        do_nothing_rho : float
            The max. rho obtained by simulating a do-nothing action.
        action_rho : float
            The max. rho obtained by simulating the selected action.
        duration
            The time in (ms) that the strategy required to select the action.

        Returns
        -------
            dict
                The dictionary representing the datapoint.
        """
        datapoint = {
            'action_index': action_index,
            'do_nothing_rho': do_nothing_rho,
            'action_rho': action_rho,
            'duration': duration,
            'timestep': obs.current_step,
            'observation_vector': obs.to_vect()
        }
        return datapoint

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


class IdleStrategy(AgentStrategy):
    """
    Strategy that produces only do-nothing actions.
    """

    def __init__(self, do_nothing_action: grid2op.Action.BaseAction, suppress_warning: bool = False):
        """
        Parameters
        ----------
        do_nothing_action : grid2op.Action.BaseAction
            The do-nothing action
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """
        super()
        self.do_nothing_action = do_nothing_action

        if not suppress_warning:
            warnings.warn("\nSaving actions as datapoints is not supported by class IdleStrategy; " +
                          "\nthe second output of the select_action method is always None.", stacklevel=2)

    def select_action(self, observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, None]:
        """
        Selects an action.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        grid2op.Action.BaseAction
            The do-nothing action.
        None
            An optional dictionary which contains information about the action. Not provided by this subclass.
        """
        return self.do_nothing_action, None


class GreedyStrategy(AgentStrategy):

    def __init__(self,
                 do_nothing_threshold: float,
                 do_nothing_action: grid2op.Action.BaseAction,
                 reduced_action_list: Sequence[grid2op.Action.TopologyAction],
                 suppress_warning: bool = False):
        """
        Parameters
        ----------
        do_nothing_threshold : float
            The threshold below which do-nothing actions are always selected and no datapoints are returned.
        do_nothing_action : grid2op.Action.BaseAction
            The do-nothing action.
        reduced_action_list : Sequence[grid2op.Action.TopologyAction]
            The reduced action list, containing the actions simulated and selected from by the greedy agent.
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """
        super()
        self.do_nothing_threshold = do_nothing_threshold
        self.do_nothing_action = do_nothing_action
        self.reduced_action_list = reduced_action_list

        if not suppress_warning:
            warnings.warn("\nSaving inference durations in datapoints is not implemented; " +
                          "\nthe value in datapoint_dict is always 0.", stacklevel=2)

    def select_action(self,
                      observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, Optional[dict]]:
        """
        Selects an action based on the greedy strategy: the action that minimizes the max. rho in the simulated
        next timestep is selected.

        Parameters
        ----------
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        datapoint_dict : Optional[dict]
            A dictionary which contains information about the action, or None. None is returned if the datapoint should
            not be saved as action.
        """
        if observation.rho.max() > self.do_nothing_threshold:
            best_action = self.do_nothing_action
            best_action_index = -1
            do_nothing_rho = best_rho = self.get_max_rho_simulated(observation, best_action)

            # Simulate each action
            for index, action in enumerate(self.reduced_action_list):

                # Skip any action that tries to change a line status
                if (not action._lines_impacted is None) and sum(action._lines_impacted) > 0:
                    continue

                # Obtain the max. rho of the observation resulting from the simulated action
                action_rho = self.get_max_rho_simulated(observation, action)

                # If an action results in the lowest max. rho so far, store it as the best action so far
                if action_rho < best_rho:
                    best_rho = action_rho
                    best_action = action
                    best_action_index = index

            action = best_action

            return action, self.create_datapoint_dict(best_action_index, observation, do_nothing_rho, best_rho)
        else:
            return self.do_nothing_action, None


class NMinusOneStrategy(AgentStrategy):

    def __init__(self,
                 do_nothing_threshold: float,
                 action_space: grid2op.Action.ActionSpace,
                 reduced_action_list: Sequence[grid2op.Action.TopologyAction],
                 line_outages_to_consider: Sequence[int],
                 N0_rho_threshold: float,
                 suppress_warning: bool = False):
        """

        Parameters
        ----------
        do_nothing_threshold : float
            The threshold below which do-nothing actions are always selected and no datapoints are returned.
        action_space : grid2op.Action.ActionSpace
            The action space of the environment in which the agent operates.
        reduced_action_list : Sequence[grid2op.Action.TopologyAction]
            The reduced action list, containing the actions simulated and selected from by the agent.
        line_outages_to_consider : Sequence[int]
            The lines outages considered by the agent in it's N-1 max. max. rho simulation.
        N0_rho_threshold : float
            Actions that lead to a good N0 scenario (i.e. a scenario where no line outages are considered) are
            first evaluated on their N-1 performance. This value determines what counts as 'good' N0 performance:
            if an action has a N0 max. rho that exceeds it, that actions' N-1 max. max. rho is not evaluated.
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """
        super()
        self.do_nothing_threshold = do_nothing_threshold
        self.action_space = action_space
        self.reduced_action_list = reduced_action_list
        self.line_outages_to_consider = line_outages_to_consider
        self.N0_rho_threshold = N0_rho_threshold

        if not suppress_warning:
            warnings.warn("\nSaving inference durations in datapoints is not implemented; " +
                          "\nthe value in datapoint_dict is always 0.", stacklevel=2)

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
            combined_action = self.action_space({"set_line_status": (line_idx, -1),
                                                 "set_bus": set_bus})

            # Simulate the action to obtain the max. rho, add it to the list
            max_rhos.append(self.get_max_rho_simulated(observation, combined_action))

        return max(max_rhos)

    def select_action(self,
                      observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, Optional[dict]]:
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
        observation :  grid2op.Observation.CompleteObservation
            The observation, on which to base the action.

        Returns
        -------
        action_chosen : grid2op.Action.BaseAction
            The selected action.
        datapoint_dict : Optional[dict]
            A dictionary which contains information about the action, or None. None is returned if the datapoint should
            not be saved as action.
        """
        if observation.rho.max() > self.do_nothing_threshold:
            action_chosen = sel_rho = action_idx = None
            dn_rho = self.get_max_rho_simulated(observation, self.action_space({}))

            # Predicting N-1 networks with the N-1 agent is not implemented, so select a do-nothing action when
            # a line is disabled
            if not all(observation.line_status):
                return self.action_space({}), None

            # Select the do-something actions
            actions = [(idx, a) for idx, a in enumerate(self.reduced_action_list)
                       if not self.is_do_nothing_set_bus(observation.topo_vect, a.set_bus)]
            # Add back singular do-nothing action at the start
            actions.insert(0, (-1, self.action_space({})))
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
            assert action_chosen is None or sel_rho < self.N0_rho_threshold, \
                "At this point, action chosen should be None or the the sel_rho below the threshold."

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
            assert not (action_chosen is None or sel_rho is None or action_idx is None), \
                "One of the output variables is None."
            assert sel_rho >= 0, "Sel_rho cannot be negative"
            assert len(self.reduced_action_list) > action_idx >= -1, "Action idx is outside of it's possible range."
            assert action_idx == -1 if sel_rho == np.inf else True, \
                "If sel_rho is infinite, the action should be do_nothing."

            return action_chosen, self.create_datapoint_dict(action_idx, observation, dn_rho, sel_rho)
        else:
            return self.action_space({}), None


class NaiveStrategy(AgentStrategy):
    """
    Naive strategy that simply selects the action predicted by the ML model.
    """

    def __init__(self,
                 model: Model,
                 feature_statistics: dict,
                 action_space: grid2op.Action.ActionSpace,
                 dn_threshold: float,
                 suppress_warning: bool = False):
        """
        Parameters
        ----------
        model : model.Model
            The machine learning model that predicts actions.
        feature_statistics : dict
            Dictionary with information (mean, std) per object type used to normalize features.
        action_space : grid2op.Action.ActionSpace
            The action space of the environment in which the agent operates.
        dn_threshold : float
            The threshold below which do-nothing actions are always selected and no datapoints are returned.
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """

        super()
        self.model = model
        self.feature_statistics = feature_statistics
        self.action_space = action_space
        self.dn_threshold = dn_threshold

        # Initialize action space cache used for
        config = get_config()
        lout_considered = config['training']['settings']['line_outages_considered']
        self.as_cache = postprocessing.ActSpaceCache(line_outages_considered=lout_considered)

        if not suppress_warning:
            warnings.warn("\nSaving actions as datapoints is not supported by class NaiveStrategy; " +
                          "\nthe second output of the select_action method is always None.", stacklevel=2)

    def select_action(self, observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, None]:
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
        None
            An optional dictionary which contains information about the action. Not provided by this subclass.
        """
        if observation.rho.max() > self.dn_threshold:
            P = ML_predict_obs(self.model, observation, self.feature_statistics, self.as_cache)
            P = P.numpy().astype(int)

            action = self.action_space({'change_bus': np.where(P)[0]})
        else:
            action = self.action_space({})

        return action, None


class VerifyStrategy(AgentStrategy):
    """
    Strategy that selects the action predicted by the ML model, but might reject it the action increases the max. rho
    over a threshold.
    """

    def __init__(self,
                 model: Model,
                 feature_statistics: dict,
                 action_space: grid2op.Action.ActionSpace,
                 dn_threshold: float,
                 reject_action_threshold: float,
                 suppress_warning: bool = False):

        """
        Parameters
        ----------
        model : model.Model
            The machine learning model that predicts actions.
        feature_statistics : dict
            Dictionary with information (mean, std) per object type used to normalize features.
        action_space : grid2op.Action.ActionSpace
            The action space of the environment in which the agent operates.
        dn_threshold : float
            The threshold below which do-nothing actions are always selected and no datapoints are returned.
        reject_action_threshold : float
            The threshold, if succeeded by simulating an action, that action is rejected.
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """
        super()
        self.model = model
        self.feature_statistics = feature_statistics
        self.action_space = action_space
        self.dn_threshold = dn_threshold
        self.reject_action_threshold = reject_action_threshold

        # Initialize action space cache used for
        config = get_config()
        lout_considered = config['training']['settings']['line_outages_considered']
        self.as_cache = postprocessing.ActSpaceCache(line_outages_considered=lout_considered)

        if not suppress_warning:
            warnings.warn("\nSaving actions as datapoints is not supported by class VerifyStrategy; " +
                          "\nthe second output of the select_action method is always None.", stacklevel=2)

    def select_action(self, observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, None]:
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
        None
            An optional dictionary which contains information about the action. Not provided by this subclass.
        """
        if observation.rho.max() > self.dn_threshold:
            P = ML_predict_obs(self.model, observation, self.feature_statistics, self.as_cache)
            P = P.numpy().astype(int)

            action = self.action_space({'change_bus': np.where(P)[0]})

            # Verify the action; if failed, use a do_nothing action
            simulation_max_rho = self.get_max_rho_simulated(observation, action)
            if simulation_max_rho > self.reject_action_threshold and simulation_max_rho > observation.rho.max():
                action = self.action_space({})
        else:
            action = self.action_space({})

        return action, None


class VerifyGreedyHybridStrategy(AgentStrategy):
    """
    Strategy that selects the action selected by the ML model, with two exceptions:
    1) Actions are rejected if they increase the max. rho over a threshold.
    2) Above a certain threshold, the agent takes actions using greedy simulation.
    """

    def __init__(self,
                 model: Model,
                 feature_statistics: dict,
                 action_space: grid2op.Action.ActionSpace,
                 dn_threshold: float,
                 reject_action_threshold: float,
                 reduced_action_list: Sequence[grid2op.Action.TopologyAction],
                 switch_control_threshold: float,
                 suppress_warning: bool = False):
        """
        Parameters
        ----------
        model : model.Model
            The machine learning model that predicts actions.
        feature_statistics : dict
            Dictionary with information (mean, std) per object type used to normalize features.
        action_space : grid2op.Action.ActionSpace
            The action space of the environment in which the agent operates.
        dn_threshold : float
            The threshold below which do-nothing actions are always selected and no datapoints are returned.
        reject_action_threshold : float
            The threshold, if succeeded by simulating an action, that action is rejected.
        reduced_action_list : Sequence[grid2op.Action.TopologyAction]
            The reduced action list, containing the actions simulated and selected from by the greedy agent.
        switch_control_threshold : float
            Constant above which to switch to the greedy agent.
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """
        super()
        self.verify_strategy = VerifyStrategy(model,
                                              feature_statistics,
                                              action_space,
                                              dn_threshold,
                                              reject_action_threshold,
                                              True)
        self.greedy_strategy = GreedyStrategy(dn_threshold,
                                              action_space({}),
                                              reduced_action_list,
                                              True)
        self.switch_control_threshold = switch_control_threshold

        if not suppress_warning:
            warnings.warn("\nSaving actions as datapoints is not supported by class VerifyGreedyHybridStrategy; " +
                          "\nthe second output of the select_action method is always None.", stacklevel=2)

    def select_action(self, observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, None]:
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
        None
            An optional dictionary which contains information about the action. Not provided by this subclass.
        """
        if observation.rho.max() > self.switch_control_threshold:
            action, _ = self.greedy_strategy.select_action(observation)
            return action, None
        else:
            action, _ = self.verify_strategy.select_action(observation)
            return action, None


class VerifyNMinusOneHybridStrategy(AgentStrategy):
    """
    Strategy that selects the action selected by the ML model, with two exceptions:
    1) Actions are rejected if they increase the max. rho over a threshold.
    2) Above a certain threshold, the agent takes actions using greedy simulation.
    """

    def __init__(self,
                 model: Model,
                 feature_statistics: dict,
                 env_action_space: grid2op.Action.ActionSpace,
                 dn_threshold: float,
                 reject_action_threshold: float,
                 reduced_action_list: Sequence[grid2op.Action.TopologyAction],
                 line_outages_to_consider: list,
                 switch_control_threshold: float,
                 N0_rho_threshold: float,
                 suppress_warning: bool = False):
        """
        Parameters
        ----------
        model : model.Model
            The machine learning model that predicts actions.
        feature_statistics : dict
            Dictionary with information (mean, std) per object type used to normalize features.
        dn_threshold : float
            The threshold below which do-nothing actions are always selected and no datapoints are returned.
        reject_action_threshold : float
            The threshold, if succeeded by simulating an action, that action is rejected.
        reduced_action_list : Sequence[grid2op.Action.TopologyAction]
            The reduced action list, containing the actions simulated and selected from by the agent.
        line_outages_to_consider : Sequence[int]
            The lines outages considered by the agent in it's N-1 max. max. rho simulation.
        N0_rho_threshold : float
            N-0 rho filter for actions of the N-1 agent.
        switch_control_threshold : float
            Constant above which to switch to the greedy agent.
        suppress_warning : bool
            Whether to suppress warnings during initialization.
        """
        super()
        self.verify_strategy = VerifyStrategy(model,
                                              feature_statistics,
                                              env_action_space,
                                              dn_threshold,
                                              reject_action_threshold,
                                              True)
        self.nminusone_strategy = NMinusOneStrategy(dn_threshold,
                                                    env_action_space,
                                                    reduced_action_list,
                                                    line_outages_to_consider,
                                                    N0_rho_threshold,
                                                    suppress_warning=True)
        self.switch_control_threshold = switch_control_threshold

        if not suppress_warning:
            warnings.warn("\nSaving actions as datapoints is not supported by class VerifyNMinusOneHybridStrategy; " +
                          "\nthe second output of the select_action method is always None.", stacklevel=2)

    def select_action(self, observation: grid2op.Observation.CompleteObservation) \
            -> Tuple[grid2op.Action.BaseAction, None]:
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
        None
            An optional dictionary which contains information about the action. Not provided by this subclass.
        """
        if observation.rho.max() > self.switch_control_threshold:
            action, _ = self.nminusone_strategy.select_action(observation)
            return action, None
        else:
            action, _ = self.verify_strategy.select_action(observation)
            return action, None


def ML_predict_obs(model: training.models.Model,
                   obs: grid2op.Observation.CompleteObservation,
                   fstats: dict,
                   as_cache: postprocessing.ActSpaceCache) -> \
        torch.Tensor:
    """
    Make a model prediction from an observation.

    Parameters
    ----------
    model : training.models.Model
        The model to predict with.
    obs : grid2op.Observation.CompleteObservation
        The observation to predict from.
    fstats : dict
        The feature statistics used to normalize the environment features.
    as_cache : postprocessing.ActSpaceCache
        The action space cache, which postprocesses the prediction so that it is a valid action.

    Returns
    -------
    torch.Tensor
        The postprocessed prediction.
    """
    obs_dict = obs.to_dict()
    device = 'cpu'

    # Extract and normalize gen, load, or, ex features and topo_vect
    gen_features = extract_gen_features(obs_dict)
    norm_gen_features = (gen_features - fstats['gen']['mean']) / fstats['gen']['std']
    load_features = extract_load_features(obs_dict)
    norm_load_features = (load_features - fstats['load']['mean']) / fstats['load']['std']
    or_features = extract_or_features(obs_dict)
    norm_or_features = (or_features - fstats['line']['mean']) / fstats['line']['std']
    ex_features = extract_ex_features(obs_dict)
    norm_ex_features = (ex_features - fstats['line']['mean']) / fstats['line']['std']
    topo_vect = obs.topo_vect

    # Make a prediction with the model
    if type(model) is FCNN:
        P = _predict_FCNN(model, norm_gen_features, norm_load_features, norm_or_features, norm_ex_features,
                          topo_vect)
    elif type(model) is GCN:
        P = _predict_GCN(model, obs, norm_gen_features, norm_load_features, norm_or_features, norm_ex_features,
                         topo_vect)
    else:
        raise TypeError("Model had invalid type.")

    # Match prediction to nearest valid action or the single affected subtation
    disabled_lines = np.where(~obs.line_status)[0]
    if len(disabled_lines) > 1 or (len(disabled_lines) == 1
                                   and disabled_lines[0] not in as_cache.set_act_space_per_lo.keys()):
        # Select a single substation
        P_sub_mask, P_sub_idx = g2o_util.select_single_substation_from_topovect(P,
                                                                                obs.sub_info,
                                                                                f=lambda x:
                                                                                torch.sum(torch.clamp(x - 0.5, min=0)))
        postprocessed_P = torch.zeros_like(P)
        postprocessed_P[torch.logical_and(P_sub_mask, P > 0.5)] = 1
    else:
        # Match to the closest action
        get_cabnp = as_cache.get_change_actspace_by_nearness_pred
        postprocessed_P = get_cabnp(-1 if len(disabled_lines) == 0 else disabled_lines[0],
                                    torch.tensor(obs.topo_vect, device=device), P, device)[0]

    return postprocessed_P


def _predict_FCNN(model: FCNN,
                  norm_gen_features: np.ndarray,
                  norm_load_features: np.ndarray,
                  norm_or_features: np.ndarray,
                  norm_ex_features: np.ndarray,
                  topo_vect: np.ndarray):
    """
    Makes a prediction with a FCNN model.

    Parameters
    ----------
    model : FCNN
        The FCNN model.
    norm_gen_features : np.ndarray
        The normalized generator features.
    norm_load_features : np.ndarray
        The normalized load features.
    norm_or_features : np.ndarray
        The normalized origin features.
    norm_ex_features : np.ndarray
        The normalized extremity features.
    topo_vect : np.ndarray
        The topology vector.

    Returns
    -------
    torch.Tensor
        The prediction.
    """
    device = 'cpu'
    # Combine gen, load, or, ex features and topo_vect into a single vector
    features = torch.tensor(np.concatenate((norm_gen_features.flatten(),
                                            norm_load_features.flatten(),
                                            norm_or_features.flatten(),
                                            norm_ex_features.flatten(),
                                            topo_vect)),
                            device=device, dtype=torch.float)

    # Return prediction
    P = model(features).reshape((-1))

    return P


def _predict_GCN(model: GCN,
                 obs: grid2op.Observation.CompleteObservation,
                 norm_gen_features: np.ndarray,
                 norm_load_features: np.ndarray,
                 norm_or_features: np.ndarray,
                 norm_ex_features: np.ndarray,
                 topo_vect: np.ndarray):
    """
    Makes a prediction with a GCN model.

    Parameters
    ----------
    model : GCN
        The GCN model.
    obs : grid2op.Observation.CompleteObservation
        The observation to predict from.
    norm_gen_features : np.ndarray
        The normalized generator features.
    norm_load_features : np.ndarray
        The normalized load features.
    norm_or_features : np.ndarray
        The normalized origin features.
    norm_ex_features : np.ndarray
        The normalized extremity features.
    topo_vect : np.ndarray
        The topology vector.

    Returns
    -------
    torch.Tensor
        The prediction.
    """
    config = get_config()
    disabled_lines = np.where(~obs.line_status)[0]
    device = 'cpu'

    # Obtain the reduced env variables for this line disabled
    # TODO: this might be too slow, and should perhaps be cached
    reduced_env_vars = reduced_env_variables(disabled_lines)

    # Concatenating the pos_topo_vect variables
    reduced_pos_topo_vect = np.argsort(np.concatenate([obs.gen_pos_topo_vect,
                                                       obs.load_pos_topo_vect,
                                                       reduced_env_vars['line_or_pos_topo_vect'],
                                                       reduced_env_vars['line_ex_pos_topo_vect']]))

    if len(disabled_lines) > 0:
        dis_lines_or_tv = [config['rte_case14_realistic']['line_or_pos_topo_vect'][line] for line in disabled_lines]
        dis_lines_ex_tv = [config['rte_case14_realistic']['line_ex_pos_topo_vect'][line] for line in disabled_lines]
        dis_endpoint_tv = dis_lines_or_tv + dis_lines_ex_tv

        # noinspection PyTypeChecker
        reduced_or_features = np.delete(norm_or_features, disabled_lines, axis=0)
        # noinspection PyTypeChecker
        reduced_ex_features = np.delete(norm_ex_features, disabled_lines, axis=0)
        assert norm_or_features.shape == norm_ex_features.shape, ("Origin and extremities features should have the "
                                                                  "same shape.")

        # Check and reduce the topo_vect variable
        assert all(topo_vect[dis_lines_or_tv] == -1), 'Disabled origins and extremities should be -1.'
        # noinspection PyTypeChecker
        reduced_topo_vect = np.delete(topo_vect, dis_endpoint_tv)
    else:
        reduced_or_features = norm_or_features
        reduced_ex_features = norm_ex_features
        reduced_topo_vect = obs.topo_vect

    same_busbar_e, other_busbar_e, line_e = g2o_util.connectivity_matrices(reduced_env_vars['sub_info'],
                                                                           reduced_topo_vect,
                                                                           reduced_env_vars['line_or_pos_topo_vect'],
                                                                           reduced_env_vars['line_ex_pos_topo_vect'])

    if model.network_type == NetworkType.HOMO:
        # noinspection PyTypeChecker
        edges = torch.tensor(np.append(same_busbar_e, line_e, axis=1), device=device, dtype=torch.long)
    elif model.network_type == NetworkType.HETERO:
        edges = {('object', 'line', 'object'): torch.tensor(line_e, device=device, dtype=torch.long),
                 ('object', 'same_busbar', 'object'):
                     torch.tensor(same_busbar_e, device=device, dtype=torch.long),
                 ('object', 'other_busbar', 'object'):
                     torch.tensor(other_busbar_e, device=device, dtype=torch.long)}
    else:
        raise ValueError('Invalid network_type value.')

    norm_gen_features = torch.tensor(norm_gen_features, dtype=torch.float)
    norm_load_features = torch.tensor(norm_load_features, dtype=torch.float)
    reduced_or_features = torch.tensor(reduced_or_features, dtype=torch.float)
    reduced_ex_features = torch.tensor(reduced_ex_features, dtype=torch.float)

    # Make a prediction
    P = model(norm_gen_features,
              norm_load_features,
              reduced_or_features,
              reduced_ex_features,
              edges,
              reduced_pos_topo_vect).reshape((-1))

    # Unreduce prediction
    if len(disabled_lines) > 0:
        # Adjust the indices to account for index shifting when inserting multiple elements
        add_idxs = dis_endpoint_tv.copy()
        add_idxs.sort()
        add_idxs = [idx - i for i, idx in enumerate(add_idxs)]

        # Unreduce set_action
        P = P.detach().numpy()
        P = np.insert(P, add_idxs, 0)
        P = torch.tensor(P, device=device)
        assert torch.count_nonzero(P[dis_endpoint_tv,]) == 0, \
            "Disabled line endpoints must be zero in the topo_vect after unreducing."

    return P

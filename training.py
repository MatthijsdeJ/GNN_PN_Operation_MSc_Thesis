#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:47:34 2022

@author: matthijs
"""
from typing import Sequence, List, Callable, Tuple, Optional
import torch
import metrics
from action_space.generate_action_space import get_env_actions
import wandb
from model import GCN
from dataloader import TutorDataLoader
from tqdm import tqdm
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import util

def BCELoss_labels_weighted(P: torch.Tensor, Y: torch.Tensor, W: torch.Tensor)\
        -> torch.Tensor:
    '''
    Binary cross entropy loss which allows for different weights for different labels.

    Parameters
    ----------
    P : torch.Tensor
        The predicted labels.
    Y : torch.Tensor
        The true labels.
    W : torch.Tensor
        The weights per label.

    Returns
    -------
    loss : torch.Tensor
        Tensor object of size (1,1) containing the loss value.
    '''
    P = torch.clamp(P,min=1e-7,max=1-1e-7)
    bce = W * (- Y * torch.log(P) - (1 - Y) * torch.log(1 - P))
    loss = torch.mean(bce)
    return loss

def tv_groupby_subst(tv: Sequence, sub_info: Sequence[int]) -> \
        List[Sequence]:
    '''
    Group a sequence the shape of the topology vector by the substations.

    Parameters
    ----------
    tv : Sequence
        Sequence the shape of the topology vector.
    sub_info : Sequence[int]
        Sequence with elements containing the number of object connected to 
        each substation.

    Returns
    -------
    List[Sequence]
        List, each element corresponding to a Sequence of objects in tv that
        belong to a particular substation.
    '''
    i = 0
    gs = []
    for ss in sub_info:
        gs.append(tv[i:i+ss])
        i+=ss
    return gs

def argmax_f(x: Sequence, f: Callable) -> int:
    '''
    Take the argmax (i.e. the index) based on a the maximum of a particular function.

    Parameters
    ----------
    x : Sequence
        The sequence to find the argmax of.
    f : Callable
        The function to apply to the elements

    Returns
    -------
    int
        The index produced by the argmax.

    '''
    return max(enumerate([f(d) for d in x]),key=lambda x:x[1])[0]

def one_hot_substation(P: Sequence, sub_info: Sequence[int]) \
        -> Tuple[torch.Tensor,Optional[int]]:
    '''
    Selects the action only at the substation for which the predictions 
    where the most extreme. Does NOT produce a one hot vector.

    Parameters
    ----------
    P : Sequence
        The predictions.
    sub_info : Sequence[int]
        Sequence with elements representing the number of object connected to 
        each substation.

    Returns
    -------
    torch.Tensor
        The tensor representing the predictions, but with zero except
        for at the most extreme substation. If all elements are below the 0.5
        threshold, zero everywhere.

    Optional[int]
        Index of the substation. None if all elements are below the 0.5
        threshold.

    '''
    if all(P<0.5):
        return torch.zeros_like(P), None
    
    P_grouped = tv_groupby_subst(P,sub_info)
    max_subsation_idx = argmax_f(P_grouped, lambda x: torch.sum(torch.clamp(x-0.5,min=0)))
    return torch.cat([sub if i==max_subsation_idx else torch.zeros_like(sub) 
                          for i,sub in enumerate(P_grouped)]), max_subsation_idx


def get_Y_subchanged(Y: Sequence, sub_info: Sequence[int])\
        -> Tuple[torch.Tensor,Optional[int]]:
    '''
    Find the substation at which the 'true' actions(i.e. the label) were taken.

    Parameters
    ----------
    Y : Sequence
        The labels.
    sub_info : Sequence[int]
        Sequence with elements representing the number of object connected to 
        each substation.

    Returns
    -------
    torch.Tensor
        The mask of the substation where the true actions are taken are 
        in the topology vector. Fully zeros if the 'true' action is a do-nothing action.
    Optional[int]
        Index of the substation. None if the 'true' action is a do-nothing action.
    '''
    if all(Y<0.5):
        return torch.zeros_like(Y),None
    
    Y_grouped = tv_groupby_subst(Y,sub_info)
    idx = argmax_f(Y_grouped, lambda x: torch.sum(x))
    return torch.cat([torch.ones_like(sub)  if i==idx else torch.zeros_like(sub) 
                          for i,sub in enumerate(Y_grouped)]), idx
        
def label_weights(mask: torch.Tensor, w: float) \
        -> torch.Tensor:
    '''
    Give the masked labels a specific weight value, and the other weights value 1.

    Parameters
    ----------
    mask : torch.Tensor[bool]
        Mask indicating which labels to give a special weight.
    w : float
        The 'special' weight value.

    Returns
    -------
    weights : torch.Tensor[float]
        The resulting label weights, consisting of the values '1' and 
        'w_when_zero'.
    '''
    weights=torch.ones_like(mask)
    weights[mask.detach().bool()] = w
    return weights

class ActSpaceCache():
    '''
    Class for storing the action spaces per line removed, so to make
    retrieving the action spaces more efficient.
    '''
    
    def __init__(self, line_outages_considered: Sequence[int] = [-1]):
        '''
        Parameters
        ----------
        line_outages_considered : Sequence[int], optional
            For which lines removed to store the action space.
            -1 in this list represent no line removed. The default is [-1].
        '''
        self.set_act_space_per_lo = {}
        for lo in line_outages_considered:
            self.set_act_space_per_lo[lo]=torch.tensor(
                [a._set_topo_vect for a in get_env_actions(lo)]) 
    
    def get_change_actspace_by_nearness_pred(self,
                                    line_disabled: int,
                                    topo_vect: torch.Tensor,
                                    P: torch.Tensor,
                                    device: torch.device) -> torch.Tensor:
        '''
        Given a prediction from the model:
            (1) compute the corresponding 'change' action space
            (2) order the corresponding 'change' action space based on the 
            nearness to the prediction.
    
        Parameters
        ----------
        line_disabled : int
            The line disabled, used to index the correct action space.
            -1 represent the action space with no line disabled.
        topo_vect : torch.Tensor[int]
            The current topoly vector. Should have elements in {1,2}.
            Used to compute the 'change' action space. Should have a length
            corresponding to the number of objects in the network.
        P : torch.Tensor[float]
            The predicions from the model. Predictions are in range (0,1). 
            Should have a length corresponding to the number of objects 
            in the network.
        device : torch.device
            What device to load the data structures on.
    
        Returns
        -------
        change_act_space : torch.Tensor
            Sorted 'change' action space. Should have the shape 
            (N_ACTIONS, N_OBJECTS). Sorted by increasing nearness to the 
            predicted action space.
        '''
        #Index the action space for 'set' actions
        set_act_space = self.set_act_space_per_lo[line_disabled].to(device)
        
        #Compute the 'change' action space
        topo_vect_rpt = topo_vect.repeat(set_act_space.shape[0],1)
        change_act_space = (set_act_space != 0) * (set_act_space!=topo_vect_rpt)
    
        #Remove all do-nothing actions
        change_act_space = change_act_space[change_act_space.sum(dim=1) != 0]
        
        #Add back a single do-nothing action
        change_act_space = torch.cat([torch.zeros((1,change_act_space.shape[1]),
                                                  device=device),
                                      change_act_space])
        
        #Calculate the row-wise abs. difference between the predicted action
        #and the 'change' action space
        P_rpt = P.repeat(change_act_space.shape[0],1)
        P_diff = abs(P_rpt - change_act_space)
        
        #Sort the 'change' action space by the difference with the 
        #predicted action
        P_diff_ind_sorted = torch.argsort(P_diff.sum(axis=1))
        change_act_space = change_act_space[P_diff_ind_sorted]
        
        return change_act_space
    
class Run():
    '''
    Class that specifies the running of the GCN model.
    '''
    
    def __init__(self, 
                 config: dict):
        '''
        Parameters
        ----------
        config : dict
            Dictionary functioning as the config. Should have the entries as 
            in config.yaml .
        '''
    
        #Save some configurations
        self.config = config
        self.train_config = train_config = config['training']
        processed_data_path = config['paths']['processed_tutor_imitation']
        matrix_cache_path = config['paths']['con_matrix_cache']
        feature_statistics_path = config['paths']['feature_statistics']
        
        #Specify device to use
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                   else 'cpu')
        
        #Init model
        self.model = GCN(train_config['hyperparams']['LReLu_neg_slope'],
                         train_config['hyperparams']['weight_init_std'],
                         train_config['constants']['N_f_gen'],
                         train_config['constants']['N_f_load'],
                         train_config['constants']['N_f_endpoint'],
                         train_config['hyperparams']['N_GNN_layers'],
                         train_config['hyperparams']['N_node_hidden'],
                         train_config['hyperparams']['aggr'],
                         train_config['hyperparams']['network_type'])
        self.model.to(self.device)
#        if train_config['hyperparams']['kaiming_weight_init']:
#            self.model.init_weights_kaiming()

        #Init optimizer
        w_decay = train_config['hyperparams']['weight_decay']
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=train_config['hyperparams']['lr'],
                                          weight_decay=w_decay)
        
        #Initalize dataloaders
        network_type = train_config['hyperparams']['network_type']
        self.train_dl = TutorDataLoader(processed_data_path + '/train', 
                                        matrix_cache_path, 
                                        feature_statistics_path,
                                        device=self.device,
                                        network_type=network_type,
                                        train=True)
        self.val_dl = TutorDataLoader(processed_data_path + '/val', 
                                      matrix_cache_path, 
                                      feature_statistics_path,
                                      device=self.device,
                                      network_type=network_type,
                                      train=False)
        
        #Initialize metrics objects
        IA = metrics.IncrementalAverage
        metrics_dict = {
            'macro_accuracy': (metrics.macro_accuracy,IA()),
            'micro_accuracy': (metrics.micro_accuracy,IA()),
            'n_predicted_changes': (metrics.n_predicted_changes, IA()),
            'any_predicted_changes': (metrics.any_predicted_changes, IA()),
            'macro_accuracy_one_sub': (metrics.macro_accuracy_one_sub, IA()),
            'micro_accuracy_one_sub': (metrics.micro_accuracy_one_sub, IA()),
            'train_loss': (lambda **kwargs: kwargs['l'],IA()),
            'accuracy_predicted_substation': 
                (metrics.accuracy_predicted_substation, IA())
        }
        train_metrics_dict = dict([('train_'+k,v) for 
                                   k,v in metrics_dict.items()])
        val_metrics_dict = dict([('val_'+k,v) for 
                                 k,v in metrics_dict.items()])
        val_metrics_dict['val_macro_accuracy_valid'] = \
            (metrics.macro_accuracy_valid,IA())
        val_metrics_dict['val_micro_accuracy_valid'] = \
            (metrics.micro_accuracy_valid,IA())
        IAM = metrics.IncrementalAverageMetrics
        self.train_metrics = IAM(train_metrics_dict)
        self.val_metrics = IAM(val_metrics_dict)

        #Initialize action space cache used for 
        self.as_cache = ActSpaceCache()

        #Start wandb run
        self.run = wandb.init(project=train_config['wandb']["project"],
                              entity=train_config['wandb']["entity"],
#                              name=train_config['wandb']['model_name'],
                              tags=train_config['wandb']['model_tags'],
                              config=train_config)
        self.run.watch(self.model,
                       log_freq=train_config['settings']['train_log_freq'],
                       log='all',
                       log_graph=True)
        
    def predict_datapoint(self, dp: dict) -> torch.Tensor:
        '''
        Extract the necessary information from a datapoint, and use it to
        make a prediction from the model.

        Parameters
        ----------
        dp : dict
            The datapoint.

        Returns
        -------
        P : torch.Tensor[float]
            The prediction of the model. Should have a length corresponding to
            the number of objects in the environment. All elements should be in
            range (0,1).
        '''
        #Extract features
        X_gen = dp['gen_features']
        X_load = dp['load_features']
        X_or = dp['or_features']
        X_ex = dp['ex_features']   
        
        #Extract the position topology vector, which relates the
        #objects ordered by type to their position in the topology vector
        object_ptv = dp['object_ptv']
        
        #Extract the edges
        E = dp['edges']
        
        #Pass through the model
        P = self.model(X_gen, X_load, X_or, X_ex, E, object_ptv).reshape((-1))
        
        return P
        
    def process_single_train_dp(self, dp: dict, step: int):
        '''
        Process a single training datapoint. This involves:
            (1) Making a model predictiong
            (2) Extracting the label and smoothing it
            (3) Computing the weighted loss
            (4) Updating the gradients
            (5) Possibly, updating the model weights and resetting gradients
            (6) Updating the training metrics
            
        Parameters
        ----------
        dp : dict
            The datapoint.
        step : int
            The current step in the run.
        '''
        #Make model prediction
        P = self.predict_datapoint(dp)   

        #Extract the label, apply label smoothing
        Y = dp['change_topo_vect']
        label_smth_alpha = self.train_config['hyperparams'] \
                                            ['label_smoothing_alpha']
        Y_smth =(1-label_smth_alpha)*dp['change_topo_vect'] + \
                  label_smth_alpha*0.5*torch.ones_like(Y,device=self.device)
             
        #Compute the weights for the loss
        non_sub_label_weight = self.train_config['hyperparams'] \
                                            ['non_sub_label_weight']
        Y_sub_mask, Y_sub_idx = get_Y_subchanged(Y,dp['sub_info'])
        weights = label_weights(1-Y_sub_mask,non_sub_label_weight)
        
        #Compute the loss, update gradients
        l = BCELoss_labels_weighted(P,Y_smth,weights)
        l.backward()
        
        #If the batch is filled, update the model, reset gradients
        batch_size = self.train_config['hyperparams']['batch_size']
        if (not step%batch_size) and (step!=0):
            self.optimizer.step()
            self.model.zero_grad()
            
        #Update metrics
        one_sub_P, P_subchanged_idx = one_hot_substation(P,dp['sub_info'])
        self.train_metrics.log(P=P,Y=Y,one_sub_P=one_sub_P,l=l,
                               P_subchanged_idx=P_subchanged_idx,
                               Y_subchanged_idx=Y_sub_idx)
        
    def process_single_val_dp(self, dp: dict, step: int) \
        -> Tuple[torch.Tensor, torch.tensor, int, torch.tensor, int,
                 torch.Tensor]:
        '''
        Process a single validation datapoint. This involves:
            (1) Making a model prediction
            (2) Extracting the label and smoothing it
            (3) Computing the weighted loss
            (4) Updating the validation metrics
            (5) Returning statistics for further analysis
            
        Parameters
        ----------
        dp : dict
            The datapoint.
        step : int
            The current step in the run.

        Returns
        -------
        y : torch.Tensor[int]
            The label. Should have the lenght equal to the number of objects in
            the network. Elements should be in {0,1}.
        nearest_valid_P : torch.Tensor[int]
            The valid action nearest to the prediction. Should have the length 
            equal to the number of objects in the network. Elements should be 
            in {0,1}.
        Y_sub_idx : int
            The index of substation changed in the label. 
        Y_sub_mask : torch.Tensor[bool]
            The mask indiciating which objects in the topology vector 
            correspond to the true changed substation. Should have the length 
            equal to the number of objects in the network. Elements should be 
            in {0,1}.
        P_subchanged_idx : int
            The index of substation changed in the predicted action. Computed
            based on nearest_valid_P.
        nearest_valid_actions : torch.Tensor[int]
            Matrix indicating the order of the valid actions in order of 
            nearness to the prediction actions. Rows indicate actions. 
            Should have dimensions (n_actions, n_objects).
        '''

        #Make model prediction
        P = self.predict_datapoint(dp)  
        
        #Extract the label, apply label smoothing
        Y = dp['change_topo_vect']
        label_smth_alpha = self.train_config['hyperparams']\
                                            ['label_smoothing_alpha']
        Y_smth =(1-label_smth_alpha)*dp['change_topo_vect'] + \
                  label_smth_alpha*0.5*torch.ones_like(Y,device=self.device)

        #Compute the weights for the loss
        non_sub_label_weight = self.train_config['hyperparams'] \
                                            ['non_sub_label_weight']
        Y_sub_mask, Y_sub_idx = get_Y_subchanged(Y,dp['sub_info'])
        weights = label_weights(1-Y_sub_mask,non_sub_label_weight)
        
        #Compute the loss, update gradients
        l = BCELoss_labels_weighted(P,Y_smth,weights)

        #Calculate statistics for metrics
        one_sub_P, P_subchanged_idx = one_hot_substation(P,dp['sub_info'])
        
        get_cabnp = self.as_cache.get_change_actspace_by_nearness_pred
        nearest_valid_actions = get_cabnp(dp['line_disabled'],
                                          dp['topo_vect'],
                                          P, 
                                          self.device)
        nearest_valid_P = nearest_valid_actions[0]
        _, P_subchanged_idx = one_hot_substation(nearest_valid_P,
                                                 dp['sub_info'])
        
        #Update metrics
        self.val_metrics.log(P=P,Y=Y,one_sub_P=one_sub_P,l=l,
                             P_subchanged_idx=P_subchanged_idx,
                             Y_subchanged_idx=Y_sub_idx, 
                             nearest_valid_P=nearest_valid_P)

        #Return statistics used in further analysis
        return Y, nearest_valid_P, Y_sub_idx, Y_sub_mask, P_subchanged_idx, \
               nearest_valid_actions
    
    def evaluate_val_set(self, step: int, run: wandb.sdk.wandb_run.Run):
        '''
        Evaluate the validation set. Consists of:
            (1) updating validation metrics,
            (2) creating a confusion matrix for the substations, 
            (3) creating a histogram of the ranks of the true actions in the 
            list of valid actions sorted by nearness to the predicted actions,
            (4) creating histograms of the difference between self weights and
            other weights,
            (5) creating a stacked histogram of the labels and the (in)correct
            classifications of those.
        All of these are logged to a wandb run.

        Parameters
        ----------
        step : int
            The current step.
        run : wandb.sdk.wandb_run.Run
            The wandb run to log the analysis results to.
        '''
        #Initializing lists for tracking the predicted/true substations
        Y_subs = []
        P_subs = []
        
        #Initializing lists for tracking the ranks of the true actions in 
        #the list of valid actions sorted by nearness to the predicted actions
        Y_rank_in_nearest_v_acts = []
        
        #Initializing counters for counting the number of (in)correct
        #classifications of each label
        correct_counter = collections.Counter()
        wrong_counter = collections.Counter()
        
        with torch.no_grad():
            for dp in self.val_dl:
                Y, nearest_valid_P, Y_sub_idx, Y_sub_mask, P_subchanged_idx, \
                nearest_valid_actions = self.process_single_val_dp(dp, step)
                    
                if not self.config['training']['settings']['advanced_val_analysis']:
                    continue
                
                #Increment the counters for counting the number of (in)correct
                #classifications of each label
                sub_Y = tuple(Y[Y_sub_mask.bool()].cpu().tolist())
                if torch.equal(nearest_valid_P,torch.round(Y)):
                    correct_counter[(Y_sub_idx,sub_Y)] += 1
                else:
                    wrong_counter[(Y_sub_idx,sub_Y)] += 1
    
                #Update lists for tracking the predicted/true substations
                Y_subs.append(Y_sub_idx)
                P_subs.append(P_subchanged_idx)
                
                #Update lists for tracking the ranks of the true actions in 
                #the list of valid actions sorted by nearness to the predicted 
                #actions
                Y_index_in_valid = torch.where((nearest_valid_actions == Y) \
                                               .all(dim=1))[0].item()
                Y_rank_in_nearest_v_acts.append(Y_index_in_valid)
            
            #Logging metrics
            self.val_metrics.log_to_wandb(run, step)
            self.val_metrics.reset()
            
            if not self.config['training']['settings']['advanced_val_analysis']:
                return
            
            #Logging substation confusion matrix
            Y_subs = [(v if v is not None else -1) for v in Y_subs]
            P_subs = [(v if v is not None else -1) for v in P_subs]
            n_subs = self.config['rte_case14_realistic']['n_subs']
            classes=np.arange(-1,n_subs).tolist()
            disp = ConfusionMatrixDisplay.from_predictions(Y_subs,
                                                           P_subs,
                                                           labels=classes)
            fig = disp.figure_
            fig.set_size_inches(12, 12)
            run.log({"sub_conf_mat":fig},step=step)
            plt.close(fig)
            
            #Logging histogram of the ranks of the true actions in the list of 
            #valid actions sorted by nearness to the predicted actions
            run.log({"Y_rank_in_nearest_v_acts": 
                     wandb.Histogram(Y_rank_in_nearest_v_acts)},step=step)
            
            #Logging difference between the self weights and the other weights
            diffs = self.model.compute_difference_weights()
            diffs = dict([('diffs_weights_'+k,v) for k,v in diffs.items()])
            run.log(dict([(k,wandb.Histogram(v)) for k,v in diffs.items()]),
                    step)
            
            #Logging action counters as stacked histogram
            comb_counter_mc = (correct_counter+wrong_counter).most_common()
            correct_indices = util.flatten([correct_counter[a]*[i] for i,(a,_)\
                    in enumerate(comb_counter_mc)])
            wrong_indices = util.flatten([wrong_counter[a]*[i] for i,(a,_) \
                    in enumerate(comb_counter_mc)])
            fig, ax = plt.subplots()
            n_bins = max(correct_indices + wrong_indices)
            ax.hist([correct_indices,wrong_indices], bins=n_bins,stacked=True)
            run.log({"action_correct_dist": fig},step=step) 
            plt.close(fig)
        
        
    def start(self):
        '''
        Start the training run. Includes periodic evaluation on the validation
        set.
        '''
        with self.run as run:
            
            #Initalize progress bar
            n_epoch = self.train_config['hyperparams']['n_epoch']
            est_tsize = self.train_config['constants']['estimated_train_size']
            pbar = tqdm(total=n_epoch*est_tsize)
        
            self.model.train()
            self.model.zero_grad()
            step=0
            
            for e in range(n_epoch):
                for dp in self.train_dl:
                    #Process a single train datapoint
                    self.process_single_train_dp(dp, step)

                    #Periodically log train metrics
                    train_log_freq = self.train_config['settings'] \
                                                      ['train_log_freq']
                    if (not step%train_log_freq) and (step!=0):
                        self.train_metrics.log_to_wandb(run,step)
                        self.train_metrics.reset()
                    
                    #Periodically evaluate the validation set
                    val_log_freq = self.train_config['settings'] \
                                                    ['val_log_freq']
                    if (not step%val_log_freq) and (step!=0):
                          self.model.eval()
                          self.evaluate_val_set(step, run)
                          self.model.train()
                          
                    step+=1
                    pbar.update(1)
            pbar.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:11:12 2021

@author: matthijs
"""
from typing import Optional, Dict, Callable, List, Tuple

class IncrementalAverage():
    '''
    A class for keeping an average that can be incremented one value at the time.
    '''
    def __init__(self):
        self.sum=0.0
        self.n=0
        
    def log(self,val: Optional[float]):
        '''
        Increment the average with a value.

        Parameters
        ----------
        val : Optional[float]
            The value. Can be 'None', and it will then be ignored.
        '''
        if val is not None:
            self.sum+=val
            self.n+=1
        
    def get(self) -> float:
        '''
        Get the value of the average.
        '''
        return self.sum/self.n
    
    def reset(self):
        '''
        Reset the average.
        '''
        self.sum=0.0
        self.n=0    
        
class IncrementalAverageMetrics():
    '''
    A class for storing, updating, retrieving, printing, and resetting 
    multiple incremental averages.
    '''
    
    def __init__(self, metrics_dict: Dict[str,Tuple[Callable,IncrementalAverage]]):
        '''
        Parameters
        ----------
        metrics_dict : dict
            The dictionary is of the typeDict[str,tuple(Callable,IncrementalAverage)].
            The string is the metric name.
            The 'Callable' is the function that computes the metric.
            The 'IncrementalAverage' is, suprisingly, the incremental average.
        '''
        self.metrics_dict = metrics_dict
        
    def log(self,**kwargs: dict):
        '''
        Update the incremental averages.

        Parameters
        ----------
        **kwargs : dict
            This dictionary contains arbitrary information that the metrics
            might need to update.
        '''
        [run_avr.log(f(**kwargs)) for f,run_avr in self.metrics_dict.values()]
        
    def get_values(self) -> List[Tuple[str,float]]:
        '''
        Returns
        -------
        List[Tuple[str,float]]
            List over the different metrics consisting of tuples.
            First element of each tuple represents the name of the metric,
            the second its average value.

        '''
        return [(key,val[1].get()) for key,val in self.metrics_dict.items()]
    
    def reset(self):
        '''
        Reset all incremental averages.
        '''
        [run_avr.reset() for _,run_avr in self.metrics_dict.values()]
        
    def __str__(self):
        '''
        Returns human-readable string representation of the metrics.

        Returns
        -------
        str
            The string.
        '''
        return  '\n'.join([f'{n}: {v}' for n,v in self.get_values()])
    
def correct_whether_changes(**kwargs: dict):
    '''
    Check whether the model correctly predicted whether there were any
    changes in the imitation example.
    
    Parameters
    ----------
    kwargs['any_changes'] : bool
        Whether the imitation example had any changes.
    kwargs['out'] : torch.Tensor
        The output of the model. Elements are floats in the range (0,1).
        A value below 0.5 represents no change; above 0.5, change.
            
    Returns
    -------
    float
        Metric value. For this metric, 0 or 1.
    '''
    any_changes = kwargs['any_changes']
    out = kwargs['out']
    return any_changes == any(out>=0.5)
    
def whether_changes(**kwargs):
    '''
    Check whether the model predicted any changes.
    
    Parameters
    ----------
    kwargs['out'] : torch.Tensor
        The output of the model. Elements are floats in the range (0,1).
        A value below 0.5 represents no change; above 0.5, change.
        
    Returns
    -------
    float
        Metric value. For this metric, 0 (no changes) or 1 (changes).
    '''
    out = kwargs['out']
    return any(out>=0.5)

def accuracy_if_changes(**kwargs):
    '''
    Check the accuracy of the output assuming the imitation example included
    changes. Returns 'None' otherwise.
    
    Parameters
    ----------
    kwargs['any_changes'] : bool
        Whether the imitation example had any changes.
    kwargs['set_idxs'] : torch.Tensor
        The indexes of what objects were set (NOT changed!) in the imitation
        example.
    kwargs['set_objects'] : np.array[int]
        The busbar-object connections that have been set in the imitation 
        example, and whether that 'set' action caused a change. 
        0 represents no change; 1, change.
        The vector only represents the objects indexed by 'set_idxs'.
    kwargs['out'] : torch.Tensor
        The output of the model. Elements are floats in the range (0,1).
        A value below 0.5 represents no change; above 0.5, change.
        
    Returns
    -------
    Optional[float]
        If float, the accuracy.
    '''
    if kwargs['any_changes']:
        out = kwargs['out'][kwargs['set_idxs']].round()
        changed_objects = kwargs['set_objects']
        return float(sum(out==changed_objects))/len(out)
    else:
        return None
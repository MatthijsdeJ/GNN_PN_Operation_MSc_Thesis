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
import argparse
import imitation_generation.generation as gnr
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_nothing_capacity_threshold",  help="The threshold " +
                        "max. line rho at which the tutor takes actions.",
                        required=False, default=.97, type=float)
    parser.add_argument("--disable_line",  help="The index of the line to be disabled.",
                        required=False, default=-1, type=int)
    parser.add_argument("--start_chronic_id",  help="The chronic to start with.",
                        required=False, default=0, type=int)
    parser.add_argument("--strategy", help="The strategy to select. Should be 'Greedy' or 'CheckNMinOne'.",
                        required=False, default="CheckNMinOne")
    args = parser.parse_args()
    
    gnr.generate(args.strategy,
                 args.do_nothing_capacity_threshold,
                 args.disable_line,
                 args.start_chronic_id)

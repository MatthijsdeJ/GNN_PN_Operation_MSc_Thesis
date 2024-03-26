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
import sys
import auxiliary.config
import simulation.simulation as simulation


def main():
    auxiliary.config.parse_args_overwrite_config(sys.argv[1:])
    simulation.simulate()


if __name__ == "__main__":
    main()

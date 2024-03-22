#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:01:28 2022

@author: matthijs
"""

from training.training import Run
import sys
import auxiliary.config
# import argparse
# from auxiliary.config import overwrite_config


def main():
    auxiliary.config.parse_args_overwrite_config(sys.argv[1:])

    # Start the run
    r = Run()
    r.start()


if __name__ == "__main__":
    main()

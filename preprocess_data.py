#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:27:45 2022

@author: matthijs
"""

from data_preprocessing_analysis.data_preprocessing import process_raw_tutor_data
import sys
import auxiliary.config


def main():
    auxiliary.config.parse_args_overwrite_config(sys.argv[1:])
    # Preprocess data
    process_raw_tutor_data()


if __name__ == "__main__":
    main()

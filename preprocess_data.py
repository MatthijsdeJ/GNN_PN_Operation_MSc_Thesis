#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:27:45 2022

@author: matthijs
"""

import auxiliary.util as util
from data_preprocessing_analysis.imitation_data_preprocessing import process_raw_tutor_data, divide_files_train_val_test


def main():
    # Preprocess data
    process_raw_tutor_data()

    # Divide preprocessed data files over train, val, and test folders
    divide_files_train_val_test()


if __name__ == "__main__":
    main()

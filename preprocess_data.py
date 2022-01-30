#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:27:45 2022

@author: matthijs
"""

import auxiliary.util as util
from data_preprocessing_analysis.imitation_data_preprocessing import process_raw_tutor_data
import os
from random import shuffle
import shutil


def main():
    # Preprocess data
    config = util.load_config()
    process_raw_tutor_data(config)

    # Split into train, val, test sets
    processed_path = config['paths']['processed_tutor_imitation']

    if os.path.exists(processed_path + 'train'):
        shutil.rmtree(processed_path + 'train')
    if os.path.exists(processed_path + 'val'):
        shutil.rmtree(processed_path + 'val')
    if os.path.exists(processed_path + 'test'):
        shutil.rmtree(processed_path + 'test')

    data_files = os.listdir(processed_path)
    shuffle(data_files)

    os.mkdir(processed_path + 'train')
    os.mkdir(processed_path + 'val')
    os.mkdir(processed_path + 'test')

    train_range = config['dataset']['train_perc'] * len(data_files)
    val_range = train_range + config['dataset']['val_perc'] * len(data_files)

    for i, f in enumerate(data_files):
        if i > val_range:
            os.rename(processed_path + f, processed_path + 'test/' + f)
        elif i > train_range:
            os.rename(processed_path + f, processed_path + 'val/' + f)
        else:
            os.rename(processed_path + f, processed_path + 'train/' + f)


if __name__ == "__main__":
    main()

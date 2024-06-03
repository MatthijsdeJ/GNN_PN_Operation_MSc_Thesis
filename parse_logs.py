#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:27:45 2022

@author: matthijs
"""

from data_preprocessing_analysis.data_preprocessing import process_raw_data
import sys
import auxiliary.config
from pathlib import Path
from collections import Counter
import numpy as np


def main():
    auxiliary.config.parse_args_overwrite_config(sys.argv[1:])

    config = auxiliary.config.get_config()
    directory_path = Path(config['paths']['evaluation_log']).parent
    log_filepaths = sorted(directory_path.rglob('*'))

    running_succes_ratio = []

    for filepath in log_filepaths:

        words = ['Current chronic:', ' completed ', 'Failure of day', 'powerflow', ' Action selected. ']
        word_counter = Counter()
        with open(filepath) as file:
            for line in file:

                if 'Config' in line:
                    continue

                for word in words:
                    word_counter[word] += word in line

        if (word_counter[" completed "] + word_counter["Failure of day"]) != 0:
            success_ratio = word_counter[" completed "]/(word_counter[" completed "] + word_counter["Failure of day"])
        else:
            success_ratio = 0

        running_succes_ratio.append(success_ratio)
        running_succes_ratio = running_succes_ratio[-5:]

        print(f'{filepath.stem}: {word_counter["Current chronic:"]} scenario(s), '
              f'{word_counter[" completed "]} day(s) completed, '
              f'{word_counter["Failure of day"]} day(s) failed, '
              f'{success_ratio:0.3f} success ratio, '
              f'{word_counter["powerflow"]} diverging powerflow exception(s), '
              f'{word_counter[" Action selected. "]} action(s) taken,'
              f'running mean success ratio: {np.mean(running_succes_ratio):0.4f}, ',
              f'running std succcess ratio: {np.std(running_succes_ratio):0.4f}')


if __name__ == "__main__":
    main()
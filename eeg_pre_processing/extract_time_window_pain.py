#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_time_window_pain.py
# @Author: MoonKuma
# @Date  : 2019/3/13
# @Desc  : Designed to extract time window data for pain study in ChongQing

import os
import numpy as np
import collections
import mne
from sklearn.preprocessing import normalize
import time


# Parameters
file_path_eeg = 'data/sample_data/sample_result/pain_tfr/'
file_path_erp = 'data/sample_data/sample_result/pain_ave/'
# Hyper-parameters
baseline = (None, 0)
time_window = {'early': (0.01, 0.3), 'late': (0.25, 0.4)} # this should be decided after averaging all events
time_span = 0.01

def _get_file_list(file_path, start, end):
    # intend to be private, don't call this from outside
    file_dict = dict()  # {'sub3':'data/sample_data/sample_result/pain_ave/sub5-ave.fif'}
    files = os.listdir(file_path)
    for file_name in files:
        if file_name.startswith(start) and file_name.endswith(end):
            inner_file = os.path.join(file_path, file_name)
            key = file_name.replace(end, '')
            file_dict[key] = os.path.abspath(inner_file)
    return file_dict


def _add_col(col_names, col_key):
    # intend to be private, don't call this from outside
    if col_key not in col_names:
        col_names.append(col_key)


data_dict = dict()
col_names = list()
# read erp
file_dict = _get_file_list(file_path_erp, 'sub', '-ave.fif')
for sub_id in file_dict.keys():
    file_name = file_dict[sub_id]
    evoked_list = mne.read_evokeds(file_name)
    for event in evoked_list:
        channels = event.ch_names
        comment = event.comment
        event.apply_baseline(baseline=baseline)
        # apply time window
        for tw_name in time_window.keys():
            peak_channel, peak = event.get_peak(ch_type='eeg', tmin=time_window[tw_name][0],
                                                tmax=time_window[tw_name][1])
            # save peak latency
            key_name = 'erp_peak_' + comment + '_' + tw_name
            data_dict[sub_id][key_name] = peak
            _add_col(col_names, key_name)
            # save peak amplitude
            event_copy = event.copy()
            slice_evt = event_copy.crop(tmin=peak - time_span, tmax=peak + time_span)



# do eeg

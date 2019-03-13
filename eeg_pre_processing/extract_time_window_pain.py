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
channel_head = ['F', 'C', 'T', 'P', 'O']

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

def _average_channel_head(channel_head, channels, data_array):
    # intend to be private, don't call this from outside
    channel_group = dict()  # {'F':set(),'C':set(),'T':set(),'P':set(),'O':set()}
    result_array = None
    for head in channel_head:
        channel_group[head] = set()
    for channel in channels:
        for key in channel_group.keys():
            if str(channel).startswith(key):
                channel_group[key].add(channels.index(channel))
    for key in channel_head:
        channel_list = list(channel_group[key])
        picked_data = data_array[channel_list, :]
        picked_mean = np.mean(picked_data, axis=0).reshape(1, picked_data.shape[1])
        if result_array is None:
            result_array = picked_mean
        else:
            result_array = np.append(result_array, picked_mean, axis=0)
    return result_array

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
            mean = np.mean(slice_evt.data, axis=1).reshape(slice_evt.data.shape[0], 1)


# do eeg

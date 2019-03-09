#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File    : extract_time_window_rest.py
# @Author  : MoonKuma
# @Date    : 2019/3/9
# @Desc   : Designed to extract time window from a resting data

import os
import numpy as np
import collections
import mne
from sklearn.preprocessing import normalize
import time



eeg_data_file = 'data/sample_data/sample_result/'

subs = os.listdir(eeg_data_file)

sub = subs[0]
if not sub.endswith('-tfr.h5'):
    pass  # continue
sub_id = sub.replace('-tfr.h5','').replace('sub','')
eeg_file = 'data/sample_data/sample_result/sub4-tfr.h5'



tfrs = mne.time_frequency.read_tfrs(eeg_file)[0]
freqs = tfrs.freqs
channels = list(tfrs.ch_names)

# channel_head = ['F', 'C', 'T', 'P', 'O']
channel_head = ['F', 'C', 'T', 'P']
channel_group = dict()  #{'F':set(),'C':set(),'T':set(),'P':set(),'O':set()}
for head in channel_head:
    channel_group[head] = set()
for channel in channels:
    for key in channel_group.keys():
        if str(channel).startswith(key):
            channel_group[key].add(channels.index(channel))

data = tfrs.data
mean_data = np.mean(data, axis=2)
result_array = None
for key in channel_head:
    channel_list = list(channel_group[key])
    picked_data = mean_data[channel_list,:]
    picked_mean = np.mean(picked_data, axis=0).reshape(1,picked_data.shape[1])
    if result_array is None:
        result_array = picked_mean
    else:
        result_array = np.append(result_array, picked_mean,axis=0)

norm_result_array = normalize(result_array, axis=0)
# save them
result_dict = dict()
result_col = list()
if result_array.shape == (len(channel_head), freqs.shape[0]) == norm_result_array.shape:
    for channel in channel_head:
        for index in range(0, freqs.shape[0]):
            key = 'rest_ori_' + channel + str(freqs[index])
            if key not in result_col:
                result_col.append(key)
                result_dict[key] = result_array[channel_head.index(channel),index]
            key_norm = 'rest_norm_' + channel + str(freqs[index])
            if key_norm not in result_col:
                result_col.append(key_norm)
                result_dict[key_norm] = norm_result_array[channel_head.index(channel),index]

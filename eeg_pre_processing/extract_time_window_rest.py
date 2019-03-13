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
import traceback

def get_time_window_rest(eeg_data_file, save_path):
    subs = os.listdir(eeg_data_file)
    channel_head = ['F', 'C', 'T', 'P', 'O']
    result_col = list()
    result_full = dict()
    faild_id = list()
    for sub in subs:
        if not sub.endswith('-tfr.h5'):
            continue
        sub_id = sub.replace('-tfr.h5','').replace('sub','')
        eeg_file = os.path.join(eeg_data_file, sub)
        tfrs = mne.time_frequency.read_tfrs(eeg_file)[0]
        freqs = tfrs.freqs
        channels = list(tfrs.ch_names)
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
        # construct dict
        result_dict = dict()
        try:
            norm_result_array = normalize(result_array, axis=0) # this may failed
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
            result_dict['sub_id'] = sub_id
        except:
            print('Failed at subid=',sub_id)
            traceback.format_exc()
            faild_id.append(sub_id)
        result_full[sub] = result_dict
    pass
    result_col = sorted(result_col)
    result_col_full = list(['sub_id']) + result_col
    # write in file
    save_name = 'eeg_result_v' + str(int(time.time())%10000) + '.txt'
    save_file = os.path.join(save_path,save_name)
    with open(save_file, 'w') as file_w:
        head = ','.join(result_col_full) + '\n'
        file_w.write(head)
        for sub in result_full.keys():
            data_list = list()
            data = result_full[sub]
            for key in result_col_full:
                data_list.append(str(data.setdefault(key,'NaN')))
            str2wri = ','.join(data_list) + '\n'
            file_w.write(str2wri)


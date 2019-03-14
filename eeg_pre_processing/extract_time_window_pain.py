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
from eeg_pre_processing.methods import average_channel_head,get_file_list,save_file_dict
from sklearn.preprocessing import normalize
import time


def _add_col(col_names, col_key):
    # intend to be private, don't call this from outside
    if col_key not in col_names:
        col_names.append(col_key)

# do erp
def get_erp_pain(file_path_erp, save_path):
    # file_path_erp = 'data/sample_data/sample_result/pain_ave/'
    # save_path = '~~~~'
    # Hyper-parameters
    baseline = (None, 0)
    time_window = {'N90': (0.07, 0.11),
                   'P160': (0.14, 0.18),
                   'N220':(0.2, 0.4),
                   'P480':(0.46, 0.50)} # this should be decided after averaging all events
    time_span = 0.01
    channel_head = ['F', 'C', 'T', 'P', 'O']

    # read erp
    data_dict = dict()
    col_names = list()
    erp_file_dict = get_file_list(file_path_erp, 'sub', '-ave.fif')
    for sub_id in erp_file_dict.keys():
        sub_name = sub_id.replace('-ave.fif', '').replace('sub', '')
        data_dict[sub_id] = dict()
        file_name = erp_file_dict[sub_id]
        evoked_list = mne.read_evokeds(file_name)
        # search event
        for event in evoked_list:
            channels = event.ch_names
            comment = event.comment
            event.apply_baseline(baseline=baseline)
            # apply time window
            for tw_name in time_window.keys():
                peak_channel, peak = event.get_peak(ch_type='eeg', tmin=time_window[tw_name][0],
                                                    tmax=time_window[tw_name][1])
                # save peak latency
                key_name = 'pain_erp_peak_' + comment + '_' + tw_name
                data_dict[sub_id][key_name] = peak
                _add_col(col_names, key_name)
                # save peak amplitude
                event_copy = event.copy()
                slice_evt = event_copy.crop(tmin=peak - time_span, tmax=peak + time_span)
                data_array = np.mean(slice_evt.data, axis=1).reshape(slice_evt.data.shape[0], 1)  # mean in time window
                result_array = average_channel_head(channel_head, channels, data_array)  # avg across channels
                norm_result_array = normalize(result_array, axis=0)
                if result_array.shape[0] == len(channel_head)==norm_result_array.shape[0]:
                    for channel_index in range(0, len(channel_head)):
                        tmp_data = result_array[channel_index,0]
                        tmp_key = 'pain_erp_amplitude_orig_' + comment + '_' + tw_name + '_' + channel_head[channel_index]
                        data_dict[sub_id][tmp_key] = tmp_data
                        _add_col(col_names, tmp_key)
                        tmp_data = norm_result_array[channel_index,0]
                        tmp_key = 'pain_erp_amplitude_norm_' + comment + '_' + tw_name + '_' + channel_head[channel_index]
                        data_dict[sub_id][tmp_key] = tmp_data
                        _add_col(col_names, tmp_key)
        data_dict[sub_id]['sub_id'] = sub_name
    result_col = sorted(col_names)
    result_col_full = list(['sub_id']) + result_col
    save_name = 'pain_erp_result_v' + str(int(time.time()) % 10000) + '.txt'
    save_file = os.path.join(save_path, save_name)
    save_file_dict(save_file, result_col=result_col_full, result_dict=data_dict)


# do eeg
def get_eeg_pain(file_path, save_path):
    # file_path = 'data/sample_data/sample_result/pain_tfr/'
    # save_path = 'sadasda'
    channel_head = ['F', 'C', 'T', 'P', 'O']
    time_window = {'T0': (0., 0.2),
                    'T2': (0.2, 0.4),
                    'T4':(0.4, 0.6),
                    'T6':(0.6, 0.8)}
    baseline = (None, 0)
    data_dict = dict()
    col_names = list()
    erp_file_dict = get_file_list(file_path, 'sub', '-tfr.h5')
    for sub_id in erp_file_dict.keys():
        sub_name = sub_id.replace('-tfr.h5', '').replace('sub', '')
        data_dict[sub_id] = dict()
        file_name = erp_file_dict[sub_id]
        events = mne.time_frequency.read_tfrs(file_name)
        for event in events:
            channels = event.ch_names
            comment = event.comment
            freqs = event.freqs
            event.apply_baseline(baseline=baseline)
            for tw_name in time_window.keys():
                # make copy
                event_copy = event.copy()
                sliced_evt = event_copy.crop(tmin=time_window[tw_name][0], tmax=time_window[tw_name][1])
                # mean
                mean_data = np.mean(sliced_evt.data, axis=2)
                # average across channels
                result_array = average_channel_head(channel_head, channels, mean_data)
                # normalize
                norm_result_array = normalize(result_array, axis=0)  # this may failed
                # save in to dict()
                if result_array.shape == (len(channel_head), freqs.shape[0]) == norm_result_array.shape:
                    for channel in channel_head:
                        for index in range(0, freqs.shape[0]):
                            tmp_key = 'pain_eeg_orig_' + comment + '_' + tw_name + '_' + channel+ '_' + str(freqs[index])
                            tmp_data = result_array[channel_head.index(channel),index]
                            _add_col(col_names, tmp_key)
                            data_dict[sub_id][tmp_key] = tmp_data
                            tmp_key = 'pain_eeg_norm_' + comment + '_' + tw_name + '_' + channel+ '_' + str(freqs[index])
                            tmp_data = norm_result_array[channel_head.index(channel),index]
                            _add_col(col_names, tmp_key)
                            data_dict[sub_id][tmp_key] = tmp_data
        data_dict[sub_id]['sub_id'] = sub_name
    result_col = sorted(col_names)
    result_col_full = list(['sub_id']) + result_col
    save_name = 'pain_eeg_result_v' + str(int(time.time()) % 10000) + '.txt'
    save_file = os.path.join(save_path, save_name)
    save_file_dict(save_file, result_col=result_col_full, result_dict=data_dict)





# do eeg

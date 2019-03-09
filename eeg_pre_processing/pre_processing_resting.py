#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pre_processing_resting.py
# @Author: MoonKuma
# @Date  : 2019/3/8
# @Desc  : Pre-processing EEG resting states data from Chongqing experiment

import numpy as np
import time
from eeg_pre_processing.methods import *
import traceback


# input
# data_path = 'data/sample_data/Rest/'
# result_path_eeg = 'data/sample_data/sample_result/'
# test_num = 1
# target_file = None


def pre_processing_rest(data_path, result_path_eeg, test_num=0, target_file = None):
    # hyper parameters
    patten = 'Rest_.cnt'
    n_cycles = 2.0
    event_id = {"rest": 99}
    sample_rate = 250
    filter_eeg = (1., None)
    time_window = (-1, 1) # need to be long enough
    baseline_erp = (0, 0) # do not use baseline
    # Delta (~3 Hz), Theta(3.5~7.5 Hz), Alpha(7.5~13 Hz), Beta(14~ Hz), Gamma(28~)
    freqs = np.array([2.5, 5.0, 10.0, 17., 35.])
    # reject
    reject = 25.0
    # overwrite load func
    file_dict = get_file_dict(data_path=data_path, patten=patten)
    ICA_failed = dict()
    Morlet_failed = dict()
    # iterate
    sub_ids = list(file_dict.keys())
    # test
    if 0 < test_num < len(sub_ids):
        sub_ids = sub_ids[0: test_num]
    # target
    if target_file is not None:
        sub_ids = [target_file]
    ts_total = time.time()
    for sub_id in sub_ids:
        ts = time.time()
        msg = '====start computing : ' + sub_id
        print(msg)
        if len(file_dict[sub_id]) < 1:
            print('[Warning]', sub_id, 'is missing!')
            continue
        # concat
        raw = concat_raw_cnt(file_dict[sub_id])
        # down sample
        raw.resample(sample_rate, npad="auto")
        # filter
        raw.filter(filter_eeg[0], filter_eeg[1], fir_design='firwin')
        # set events here
        events = set_rest_events(raw_file=raw, event_id=event_id['rest'],time_window=time_window)
        print('events.shape', events.shape)
        # ICA
        try:
            ica = perform_ICA(raw)
        except:
            msg = '===ICA failed for subjects:' + sub_id
            print(msg)
            ICA_failed[sub_id] = traceback.format_exc()

        # Epoch
        eeg_evoked_list, eeg_epochs = epoch_raw(raw_copy=raw, time_window=time_window, event_id=event_id,
                                                baseline=None, reject=reject)
        # power
        try:
            powers = morlet_epochs(epochs=eeg_epochs, event_id=event_id, freqs=freqs, n_cycles=n_cycles)
            # save power
            file_name = result_path_eeg + sub_id + '-tfr.h5'
            mne.time_frequency.write_tfrs(file_name, powers, overwrite=True)

            msg = '====finish computing eeg : ' + sub_id + ' at time cost: ' + str(time.time() - ts)
            print(msg)
        except:
            msg = '===Morlet failed for subjects:' + sub_id
            print(msg)
            traceback.format_exc()
            Morlet_failed[sub_id] = traceback.format_exc()
    msg = 'Finish computing all data from : ' + str(len(sub_ids)) + ' subjects at time cost: ' + str(time.time() - ts_total)
    print(msg)
    return [ICA_failed, Morlet_failed]
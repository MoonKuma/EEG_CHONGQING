#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : eeg_pre_processing.py
# @Author: MoonKuma
# @Date  : 2019/2/12
# @Desc  : Check out mne API at : http://martinos.org/mne/index.html
# @Desc : This is specially used for this experiment

import os
import numpy as np
import time
from eeg_pre_processing.methods import *
import traceback

def pre_processing_pain(data_path, result_path_erp, result_path_eeg, test_num = 1, target_file=None, add_on=True):
    """
    Pre-processing pipeline for Chongqing EEG/ERP on pain study
    :param data_path: raw data path
    :param result_path_erp: where erp result should be saved
    :param result_path_eeg: where eeg result should be saved
    :param test_num: how many num should be tested
    :param target_file: a list to identify those that need to be run on
    :param add_on: whether this process will omit those that already been computed (this is based on the result_path_eeg)
    :return: ICA_failed, Morlet_failed, Concat_failed] those failed in ICA/Morlet/File loading
    """
    # parameters
    # data_path = 'data_sample/eeg_raw_data/subject_data/EEG_Original'
    # result_path_erp = 'data_sample/formal_dataset/sub_evoked_data/'
    # result_path_eeg = 'data_sample/formal_dataset/sub_power_data/'
    # patten = 'tb'
    #
    event_id = {"Neutral_F": 113, "Pain_F": 123, "Neutral_M": 114, "Pain_M": 124}
    patten = 'Pain_.cnt'
    patten_add_on = '-tfr.h5'
    sample_rate = 250
    stim_channel='STI 014'

    # erp
    filter_erp = (1., 50.)
    time_window_erp = (-1.0, 1.0)
    baseline_erp = (None, 0)
    # eeg
    filter_eeg = (1., 50.)
    time_window_eeg = (-1, 1.0)
    baseline_eeg = (None, 0)
    freqs = np.array([2.5, 5.0, 10.0, 17., 35.])  # as Delta (~3 Hz), Theta(3.5~7.5 Hz), Alpha(7.5~13 Hz), Beta(14~ Hz)
    n_cycles = freqs / 2.
    # reject
    reject = 25.0

    # start computing
    file_dict = get_file_dict(data_path=data_path, patten=patten)
    # subjects failed ICA/Morlet
    Concat_failed = dict()
    ICA_failed = dict()
    Morlet_failed = dict()
    # iterate
    sub_ids = list(file_dict.keys())
    if 0 < test_num < len(sub_ids):
        sub_ids = sub_ids[0: test_num]
    if target_file is not None:
        sub_ids = [target_file]
    if add_on:
        msg = 'Compute only those not existed for add_on = True'
        print(msg)
        dir_list = os.listdir(result_path_eeg)
        for file_name in dir_list:
            if file_name.endswith(patten_add_on):
                sub_rm = file_name.replace(patten_add_on,'')
                if sub_rm in sub_ids:
                    sub_ids.remove(sub_rm)
    msg = 'Final sub_ids has length of ' + str(len(sub_ids))
    print(msg)
    ts_total = time.time()
    for sub_id in sub_ids:
        ts = time.time()
        msg = '====start computing : ' + sub_id
        print(msg)
        if len(file_dict[sub_id])<1:
            print('[Warning]',sub_id, 'is missing!')
            continue
        # concat
        try:
            raw = concat_raw_cnt(file_dict[sub_id])
        except:
            msg = sub_id + 'failed in concat'
            Concat_failed[sub_id] = traceback.format_exc()
            print(msg)
            continue
        # down sample
        raw.resample(sample_rate, npad="auto")

        # filter
        raw.filter(filter_erp[0], filter_erp[1], fir_design='firwin')
        # modify events list
        events = mne.find_events(raw, stim_channel=stim_channel)
        for i in range(0, events.shape[0]):
            if events[i, 2] == 114:
                events[i, 2] = 113
            if events[i, 2] == 124:
                events[i, 2] = 123

        # ICA
        ica = None
        try:
            ica = perform_ICA(raw)
        except:
            msg = '===ICA failed for subjects:' + sub_id
            print(msg)
            ICA_failed[sub_id] = traceback.format_exc()
        # Epoch
        erp_evoked_list, erp_epochs = epoch_raw(raw_copy=raw, time_window=time_window_erp, event_id=event_id,
                                                baseline=baseline_erp, reject=reject)
        # Save evoked data
        file_name = result_path_erp + sub_id + '-ave.fif'
        mne.write_evokeds(file_name, erp_evoked_list)
        msg = '====finish computing erp : ' + sub_id
        print(msg)
        # Epoch
        eeg_evoked_list, eeg_epochs = epoch_raw(raw_copy=raw, time_window=time_window_eeg, event_id=event_id,
                                                baseline=baseline_eeg, reject=reject)
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
            Morlet_failed[sub_id] = traceback.format_exc()
    msg = 'Finish computing all data from : ' + str(len(sub_ids)) + ' subjects at time cost: ' + str(time.time() - ts_total)
    print(msg)

    return [ICA_failed, Morlet_failed, Concat_failed]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : methods.py
# @Author: MoonKuma
# @Date  : 2019/2/16
# @Desc  : mutual methods

import os
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

# # this won't suit chongqing data
# def get_file_dict(data_path, patten):
#     # data file path like : data_sample/eeg_raw_data/EEG_Original/sub2/tb1.cnt
#     # this method scan all data inside data_path(data_sample/eeg_raw_data/EEG_Original/) and start with patten('tb')
#     # and return a dict like {sub2 : data_sample/eeg_raw_data/EEG_Original/sub2/tb1.cnt}
#     raw_data_dict = dict()
#     dir_list = os.listdir(data_path)
#     for dir_name in dir_list:
#         raw_data_dict[dir_name] = list()
#         inner_file = os.path.join(data_path, dir_name)
#         raw_names = os.listdir(inner_file)
#         for raw_name in raw_names:
#             if raw_name.startswith(patten):
#                 raw_data_dict[dir_name].append(os.path.abspath(os.path.join(inner_file, raw_name)))
#     return raw_data_dict

# this is for chondqing data structure
def get_file_dict(data_path, patten):
    raw_data_dict = dict()
    dir_list = os.listdir(data_path)
    t_f_name, t_ext = os.path.splitext(patten)
    for file_name in dir_list:
        if file_name.startswith(t_f_name) and file_name.endswith(t_ext):
            inner_file = os.path.join(data_path, file_name)
            sub_index = file_name.replace(t_f_name,'').replace(t_ext,'')
            if sub_index not in raw_data_dict.keys():
                raw_data_dict[sub_index] = list()
            raw_data_dict[sub_index].append(os.path.abspath(inner_file))
    return raw_data_dict


def concat_raw_cnt(raw_file_list, preload=True, data_format='int32', eog='header', montage = mne.channels.read_montage('standard_1020') ):
    # open and transfer a list of raw data file name into one raw data
    # has to be .cnt data
    # input : raw_file_list as the list of raw cnt file names
    # return : raw (mne.raw object as a concatenation)
    raw = mne.io.read_raw_cnt(raw_file_list[0], preload=preload, data_format=data_format, eog=eog, montage=montage)
    for i in range(1,len(raw_file_list)):
        raw_tmp = mne.io.read_raw_cnt(raw_file_list[i], preload=preload, data_format=data_format, eog=eog, montage=montage)
        mne.concatenate_raws([raw, raw_tmp])
    return raw

def set_rest_events(raw_file, event_id, time_window, perc=0.5, double_blank = 0.1, remove_existing_events=True):
    """
    self-defined events for eeg rest state data

    for example, if time_window is 1 seconds, and perc=0.5, and double_blank = 0.1
    that means, the first 10% and last 10% of data will be dropped out when marking events
    meanwhile, only half of the data is considered useful

    say the total data is 6 mins = 3600 s, time_window=1s, and pec=05, then it means that
    we first removing all existing events, then we split it into larger time window of 2s
    (there will be 1800 such larger window), then in each of that 2 seconds,
    a continuously and randomly picked 1 seconds will be regarded as the event
    , of which the center will be marked as what the event_id send: '99' here
    And finally events located inside the first 360s or last 360s will be dropped

    :param raw_file: mne raw data
    :param event_id: event_id should be like 'event_id = {"rest": 99}', only the first key is used
    :param time_window: time_windows is how long each events should be considered, eg : (0.5,0.5)
    :param perc: perc means how many percentages of the full data shall be used
    :param double_blank: double_blank marks how much data should be dropped out in the beginning and end (usually messy)
    :param remove_existing_events : whether all existing events should be removed

    :return: the event list of raw data will be modified, meanwhile this will return the events list (an numpy.ndarray)
    """

    pass


def perform_ICA(raw, eeg_reject=50.):
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    ica = ICA(n_components=25, random_state=1)  # using default method 'fastica'
    reject = dict(eeg=eeg_reject)
    ica.fit(raw, picks=picks_eeg, reject=reject)
    eog_average = create_eog_epochs(raw, reject=reject, picks=picks_eeg).average()
    eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
    eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
    ica.exclude.extend(eog_inds)
    ica.apply(raw)
    return ica


def epoch_raw(raw_copy, time_window, event_id, baseline= (None, 0), reject = 10.0, stim_channel='STI 014'):
    reject = dict(eeg=reject)
    events = mne.find_events(raw_copy, stim_channel=stim_channel)
    tmin = time_window[0]  # before
    tmax = time_window[1]  # after
    baseline = baseline  # means from the first instant to t = 0
    picks_eeg = mne.pick_types(raw_copy.info, meg=False, eeg=True, eog=False, stim=False,
                               exclude=['VEOG', 'HEOG'])
    epochs = mne.Epochs(raw_copy, events, event_id, tmin, tmax, proj=True, picks=picks_eeg, baseline=baseline,
                        reject=reject, reject_by_annotation=True)
    evoked_list = list()
    for cond in event_id:
        evoked_list.append(epochs[cond].average())
    return [evoked_list, epochs]


def morlet_epochs(epochs, freqs, n_cycles, event_id, decim=1, n_jobs=1):
    # compute power
    power_list = list()
    for cond in event_id:
        power = mne.time_frequency.tfr_morlet(epochs[cond], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                              return_itc=False, decim=decim, n_jobs=n_jobs)
        power.comment = cond
        power_list.append(power)
    return power_list
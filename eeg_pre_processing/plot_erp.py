#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot_erp.py
# @Author: MoonKuma
# @Date  : 2019/3/13
# @Desc  : plot evoked data form different subjects to get a general idea of how data looks
# this is implemented though constructing a fake evoke object
# data should be identical in form

import os
import numpy as np
import collections
import mne
import time
from sklearn.preprocessing import normalize

# file_path_erp = 'data/sample_data/sample_result/pain_ave/'

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

def plot_erp(file_path_erp):
    data_dict = dict()
    count = 0
    erp_files = _get_file_list(file_path_erp, 'sub', '-ave.fif')
    for sub_id in erp_files.keys():
        file_name = erp_files[sub_id]
        evoked_list = mne.read_evokeds(file_name)
        for event in evoked_list:
            comment = event.comment
            data = event.data
            norm_data = normalize(data, axis=1)
            if comment not in data_dict.keys():
                data_dict[comment] = norm_data
            else:
                data_dict[comment] = data_dict[comment] + norm_data
        count += 1

    for event in data_dict.keys():
        data_dict[comment] = data_dict[comment]/count

    # FAKE ONE
    sub_id = list(erp_files.keys())[0]
    file_name = erp_files[sub_id]
    evoked_list = mne.read_evokeds(file_name)
    for event in evoked_list:
        comment = event.comment
        event.data = data_dict[comment]
        event.plot_joint(title=comment)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : controller.py
# @Author: MoonKuma
# @Date  : 2019/2/18
# @Desc  : control panel

import collections
import pandas as pd
from eeg_pre_processing.pre_processing_resting import pre_processing_rest
from eeg_pre_processing.pre_processing_pain import pre_processing_pain
from eeg_pre_processing.extract_time_window_rest import get_time_window_rest
from eeg_pre_processing.extract_time_window_pain import get_erp_pain,get_eeg_pain
from eeg_pre_processing.plot_erp import plot_erp
from utils.data_merge import data_merge
from eeg_random_forest.models_to_test import test_regression_model,test_classification_model
# file location
raw_data_path = 'data_sample/eeg_raw_data/EEG_Original'
result_path_erp = 'data_sample/formal_data/sub_evoked_data/'
result_path_eeg = 'data_sample/formal_data/sub_power_data/'
time_window_result = 'data_sample/formal_data/time_window_data/'
behavior_file_path = 'data_sample/formal_data/behavior_data/Behavior_raw.txt'
brain_file_path = 'data_sample/formal_data/time_window_data/time_window_data_5062.txt'
merge_data_file = 'data_sample/formal_data/merged_data/'
merge_data_name = 'data_sample/formal_data/merged_data/saving_8307.txt'
model_result_path = 'data_sample/formal_data/model_result/'


# pre-processing resting raw data
def subjects_pre_processing_resting():
    """
    Pre-processing module
    Caution this is the control panel, where parameters are written INSIDE the funcs and there won't be any return
    """
    data_path = 'G:/CQ_UsableEEG_Wenxin/CNT_rar/rest_cnt/'
    result_path_eeg = 'G:/CQ_UsableEEG_Wenxin/CNT_rar/rest_tfr/'
    test_num = 0
    target_file = ['sub496']
    ICA_failed, Morlet_failed, Concat_failed = pre_processing_rest(data_path=data_path, result_path_eeg=result_path_eeg,
                                                                   test_num=test_num,target_file=target_file)
    if len(Concat_failed.keys()) > 0:
        print('Concat failed list: ',Concat_failed.keys())
    if len(ICA_failed.keys()) > 0:
        print('ICA failed list: ',ICA_failed.keys())
    if len(Morlet_failed.keys()) > 0:
        print('Morlet failed list: ',Morlet_failed.keys())

# pre-processing pain raw data
def subjects_pre_processing_pain():
    """
        Pre-processing module
        Caution this is the control panel, where parameters are written INSIDE the funcs and there won't be any return
    """
    data_path = 'data/sample_data/Pain/'
    result_path_eeg = 'data/sample_data/sample_result/pain_tfr/'
    result_path_erp = 'data/sample_data/sample_result/pain_ave/'
    test_num = 0
    target_file = None
    ICA_failed, Morlet_failed, Concat_failed, Epoched_failed = pre_processing_pain(data_path=data_path,
                                                                                   result_path_eeg=result_path_eeg,
                                                                                   result_path_erp=result_path_erp,
                                                                                   test_num=test_num,
                                                                                   target_file=target_file)
    if len(Concat_failed.keys()) > 0:
        print('Concat failed list: ', Concat_failed.keys())
    if len(ICA_failed.keys()) > 0:
        print('ICA failed list: ', ICA_failed.keys())
    if len(Epoched_failed.keys()) > 0:
        print('Epoched failed list: ', Epoched_failed.keys())
    if len(Morlet_failed.keys()) > 0:
        print('Morlet failed list: ', Morlet_failed.keys())
    pass

# slicing time window
def time_window_rest():
    """
    Average across full time window and channel areas
    This will generate several data files in (.txt) form with time stamp for identification, yet have no return values
    """
    eeg_data_file = 'data/sample_data/sample_result/'
    eeg_time_window_save_path = 'data/sample_data/pre-processed_data/'
    get_time_window_rest(eeg_data_file=eeg_data_file, save_path=eeg_time_window_save_path)

# plot erp before deciding time window
def plot_erp():
    file_path_erp = ''
    plot_erp(file_path_erp=file_path_erp)
    pass

# time window pain erp
def time_window_pain_erp():
    """
    Get erp peak and amplitude data in certain time window for pain study
    """
    data_path = 'data/sample_data/sample_result/pain_ave/'
    save_path = 'data/sample_data/pre-processed_data/'
    get_erp_pain(file_path_erp=data_path, save_path=save_path)

def time_winodw_pain_eeg():
    """
    Get eeg data in certain time window for pain study
    """
    data_path = 'data/sample_data/sample_result/pain_tfr/'
    save_path = 'data/sample_data/pre-processed_data/'
    get_eeg_pain(file_path=data_path, save_path=save_path)



# clean behavior data

# merge them
def merge_data():
    """
    This is to merge the behavior and brain data and do some simple
    processing like one-hot gender and normalize questionnaire data
    :return:  nothing will return, see the merged data in ${merge_data_file}
    """
    data_merge(brain_file_path=brain_file_path,behavior_file_path=behavior_file_path,save_file=merge_data_file)


# testing regression models
def test_regression_models():
    full_data = pd.read_csv(merge_data_name, delimiter=',', header=0, index_col=0)
    columns = full_data.columns.values
    behavior_col = ['Gender', 'Age', 'AVGcor', 'AVGrt']
    target_col = ['Atsum']
    eeg_col = list()
    erp_col = list()
    peak_col = list()
    for i in range(0, columns.shape[0]):
        name = columns[i]
        if name.startswith('erp_peak'):
            peak_col.append(name)
        elif name.startswith('erp'):
            erp_col.append(name)
        elif name.startswith('eeg'):
            eeg_col.append(name)

    sample_dict = dict()
    sample_dict['01'] = {'x_columns': behavior_col, 'y_column': target_col}
    sample_dict['02'] = {'x_columns': peak_col, 'y_column': target_col}
    sample_dict['03'] = {'x_columns': erp_col, 'y_column': target_col}
    sample_dict['04'] = {'x_columns': eeg_col, 'y_column': target_col}
    sample_dict['05'] = {'x_columns': behavior_col + peak_col, 'y_column': target_col}
    sample_dict['06'] = {'x_columns': behavior_col + peak_col + erp_col, 'y_column': target_col}
    sample_dict['07'] = {'x_columns': behavior_col + peak_col + eeg_col, 'y_column': target_col}
    sample_dict['08'] = {'x_columns': behavior_col + peak_col + eeg_col + erp_col, 'y_column': target_col}

    test_regression_model(full_data=full_data, sample_dict=sample_dict, save_path=model_result_path+'reg_',test_times=10)

# testing classification models
def test_classification_models():
    full_data = pd.read_csv(merge_data_name, delimiter=',', header=0, index_col=0)
    columns = full_data.columns.values
    behavior_col = ['Atsum', 'Age', 'AVGcor', 'AVGrt']
    target_col = ['Gender']
    eeg_col = list()
    erp_col = list()
    peak_col = list()
    for i in range(0, columns.shape[0]):
        name = columns[i]
        if name.startswith('erp_peak'):
            peak_col.append(name)
        elif name.startswith('erp'):
            erp_col.append(name)
        elif name.startswith('eeg'):
            eeg_col.append(name)

    sample_dict = dict()
    sample_dict['01'] = {'x_columns': behavior_col, 'y_column': target_col}
    sample_dict['02'] = {'x_columns': peak_col, 'y_column': target_col}
    sample_dict['03'] = {'x_columns': erp_col, 'y_column': target_col}
    sample_dict['04'] = {'x_columns': eeg_col, 'y_column': target_col}
    sample_dict['05'] = {'x_columns': behavior_col + peak_col, 'y_column': target_col}
    sample_dict['06'] = {'x_columns': behavior_col + peak_col + erp_col, 'y_column': target_col}
    sample_dict['07'] = {'x_columns': behavior_col + peak_col + eeg_col, 'y_column': target_col}
    sample_dict['08'] = {'x_columns': behavior_col + peak_col + eeg_col + erp_col, 'y_column': target_col}

    test_classification_model(full_data=full_data, sample_dict=sample_dict, save_path=model_result_path + 'clf_',
                          test_times=10)
    pass

# subjects_pre_processing_resting()
# subjects_pre_processing_pain()
# time_window_rest()
# time_window_pain_erp()
time_winodw_pain_eeg()
# merge_data()
# test_regression_models()
# test_classification_models()
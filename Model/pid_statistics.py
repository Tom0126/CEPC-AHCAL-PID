#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 18:29
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : pid_statistics.py
# @Software: PyCharm
import os.path

import numpy as np

from Data.ReadRoot import ReadRoot
import glob
import pandas as pd

def provide_statistics_per_ep(ep_file_dir, threshold, save_dir=None):

    save_path = os.path.join(save_dir, 'pid_statistics.csv')
    if os.path.exists(save_path):
        os.system('rm {}'.format(save_path))

    statistics_dict=dict()
    ep_lists=[]
    exps = ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise']
    # statistics_dict['events_total_no']=list()
    # for exp in exps:
    #     statistics_dict[exp+'_percentage']=list()
    # statistics_dict['uncertain_{}_percentage'.format(threshold)] =list()
    ep_path_lists=glob.glob(ep_file_dir+'/**')

    stat_array=np.ones((len(exps)+2, len(ep_path_lists)))

    for index, ep_dir in enumerate(ep_path_lists):

        statistic=[0 for i in range(len(exps)+2)] # first one for all entries num, last one for uncertain one.

        for file_path in glob.glob(os.path.join(ep_dir, '*.root')):

            data=ReadRoot(file_path=file_path,tree_name='Calib_Hit',exp=exps)
            statistic[0]+=len(data.readBranch(exps[0]))
            for i, exp in enumerate (exps,start=1):
                ann_pred=data.readBranch(exp)
                statistic[i]+=len(ann_pred[ann_pred>threshold])
        statistic[-1]=statistic[0]-np.sum(statistic[1:-1])

        ep=ep_dir.replace(ep_file_dir+'/', '')
        ep_lists.append(ep)

        stat_array[0,index]=statistic[0]
        for i, exp in enumerate(exps, start=1):
            stat_array[i, index] = round(statistic[i]/statistic[0]*100,1)
        stat_array[-1, index] = round(statistic[-1]/statistic[0]*100,1)

    statistics_dict['events_total_no'] = stat_array[0]
    for i, exp in enumerate(exps, start=1):
        statistics_dict[exp + '_percentage'] = stat_array[i]
    statistics_dict['uncertain_{}_percentage'.format(threshold)] = stat_array[-1]



    df = pd.DataFrame(statistics_dict, index=ep_lists)


    df.to_csv(save_path, index_label=True, sep='\t')



if __name__ == '__main__':
    provide_statistics_per_ep(ep_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/pi',
                              save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/pi',
                              threshold=0.9)

    provide_statistics_per_ep(ep_file_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/positron',
                              save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v1/AHCAL_only/positron',
                              threshold=0.9)
    pass

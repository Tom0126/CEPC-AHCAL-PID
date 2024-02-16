#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 19:06
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : Cor.py
# @Software: PyCharm

from ReadRoot import ReadRoot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plotCorHeatmap_root(file_path,tree_name,var_list,save_path):
    data=ReadRoot(file_path=file_path,tree_name=tree_name,exp=var_list)
    data_dict={}
    for var in var_list:
        data_dict[var]=data.readBranch(var)
    data_dict = pd.DataFrame(data_dict)
    df_coor = data_dict.corr()


    plt.subplots(figsize=(9, 9), facecolor='w')

    fig = sns.heatmap(df_coor, annot=True, vmax=1, square=True, cmap="PiYG",
                      fmt='.2g')
    fig.get_figure().savefig(save_path, bbox_inches='tight', transparent=True)

def plotCorHeatmap_csv(file_path, varlist, save_path, title, **kwargs):

    df=pd.read_csv(file_path, usecols=varlist)
    df_coor=df.corr()
    df.fillna(0)
    plt.subplots(figsize=(25, 15), facecolor='w')

    if 'labels' in kwargs:
        labels=kwargs.get('labels')
    else:
        labels=varlist

    fig = sns.heatmap(df_coor, annot=True, vmax=1, vmin=-1, square=True, cmap='RdBu',
                      fmt='.2f', annot_kws={"fontsize": 25, }, xticklabels=[], yticklabels=labels)
    cbar = fig.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    # fig.set_title(title, fontsize=30)
    plt.tick_params(axis='both', labelsize=30)

    # plt.xticks(rotation=45)
    fig.get_figure().savefig(save_path, bbox_inches='tight', transparent=True)


if __name__ == '__main__':

    # var_list=['xwidth', 'ywidth', 'zwidth', 'Edep', 'shower_start' , 'shower_end', 'shower_layer', 'hit_layer',
    #           'shower_layer_ratio', 'shower_density', 'shower_length', 'shower_radius', 'FD_2D']
    # file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/jiyuan_pid/pid_Run85.root'
    # tree_name='Calib_Hit'
    # save_path='Fig/cor_run85.png'
    #
    # plotCorHeatmap(file_path=file_path,tree_name=tree_name,var_list=var_list,save_path=save_path)

    var_list=['Shower_density', 'Shower_start', 'Shower_layer_ratio', 'Shower_length', 'Hits_no', 'Shower_radius',
                  'FD_1', 'FD_6', 'Shower_end', 'layers_fired', 'Shower_layer', 'Z_width']
    labels=['Shower Density', 'Shower Start', 'Shower Layer Ratio', 'Shower Length', 'Hits Number', 'Shower Radius',
                  r'$FD_1$', r'$FD_6$', 'Shower End', 'Fired Layers', 'Shower Layers', 'Z Depth']
    plotCorHeatmap_csv(file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/mc_0720_e_pi/Train/bdt_var.csv',
                       varlist=var_list,
                       save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/mc_0720_e_pi/Train/coor_mc.png',
                       title='MC samples',
                       labels=labels)

    # plotCorHeatmap_csv(
    #     file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_0720/Train/bdt_var.csv',
    #     varlist=var_list,
    #     save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets/ckv_fd_0720/Train/coor_tb.png',
    #     title='Data samples',
    #     labels=labels
    # )
    pass

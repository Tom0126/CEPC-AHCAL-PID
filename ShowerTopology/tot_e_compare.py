#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/1 16:58
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : tot_e_compare.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from collections import Counter
from ReadRoot import ReadRoot
import numpy as np

def pick_e(file_path, tree_name, branch, threshold):


    data=ReadRoot(file_path=file_path,tree_name=tree_name,exp=[branch])
    hit_e=data.readBranch(branch)

    tot_e=[]
    for _ in hit_e:
        _=_[_>threshold]
        tot_e.append(np.sum(_))
    return np.array(tot_e)


def main(ann_tot_e_path, mc_path,save_path, mc_tree='EventTree', mc_branch='Digi_Hit_Energy',threshold=0):

    ann_tot_e=np.load(ann_tot_e_path)
    mc_tot_e=pick_e(mc_path,tree_name=mc_tree,branch=mc_branch,threshold=threshold)
    ann_mean=np.mean(ann_tot_e)
    ann_std=np.std(ann_tot_e)
    fig=plt.figure(figsize=(6,5))
    plt.hist(ann_tot_e,bins=100,density=True,label='ANN',histtype='step',color='red', range=[0,ann_mean+3*ann_std])
    plt.hist(mc_tot_e, bins=100, density=True, label='MC', histtype='step', color='black', range=[0,ann_mean+3*ann_std])
    plt.legend()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    # ep_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
    #
    # for ep in ep_list:
    #     ann_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/CheckPoint/epoch_200_lr_0.001_batch32_optim_Adam_classes_4_combinev1_1/ANA/ann_picked_pi_plus/{}GeV.npy'.format(ep)
    #     mc_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/MC/run20230428_AHCAL_Data/pi+/Digi_MC_calo_pi+_{}GeV.root'.format(ep)
    #     save_path='./Fig/{}_cp.png'.format(ep)

        # main(ann_tot_e_path=ann_path,mc_path=mc_path,save_path=save_path,)

    path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/MC/run20230428_AHCAL_Data/pi+/Digi_MC_calo_pi+_120GeV.root'
    data=ReadRoot(file_path=path,tree_name='EventTree',exp=['Hit_X','Hit_Y','Hit_Z' ] )
    _=data.readBranch('Hit_X')
    _=np.concatenate(_)
    print(Counter(_))
    _= data.readBranch('Hit_Y')
    _ = np.concatenate(_)
    print(Counter(_))
    _ = data.readBranch('Hit_Z')
    _ = np.concatenate(_)
    print(Counter(_))

    pass

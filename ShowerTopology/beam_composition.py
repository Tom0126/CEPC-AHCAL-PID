#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 11:52
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : beam_composition.py
# @Software: PyCharm



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_tot_e_purifiled_compare(raw_file_path, save_dir, ep, source, approach,stacked,ll=None, ul=None,bins=100,
                                 log=False, y_ul=None,):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    text_dict = {
        'mc': 'MC training approach',
        'tb': 'Data training approach'
    }

    color_dict = {
        'muon': 'green',
        'muon_minus': 'green',
        'muon_plus': 'green',
        'electron': 'blue',
        'electron_minus': 'blue',
        'positron': 'blue',
        'pion': 'red',
        'pion_minus': 'red',
        'pion_plus': 'red',
        'noise': 'orange',

    }

    greek_letter_dict = {
        'muon': r'Muon',
        'muon_minus': r'$' + chr(956) + '^{-}$',
        'muon_plus': r'$' + chr(956) + '^{+}$',
        'electron': r'Electron',
        'positron': r'$e^{+}$',
        'pion': r'Pion',
        'pion_minus': r'$' + chr(960) + '^{-}$',
        'pion_plus': r'$' + chr(960) + '^{+}$',
        'noise': 'Noise',
    }








    raw_file=pd.read_csv(raw_file_path, usecols=['E_dep', 'Particle_label'])

    raw_ed=raw_file['E_dep']
    raw_labels=raw_file['Particle_label']

    if ul==None:
        ul = np.mean(raw_ed) + 2.5 * np.std(raw_ed)
    if ll==None:
        ll = 0
    # low_limit=np.mean(puri_ed_) - 4 * np.std(puri_ed_)


    eds, labels, colors, weights=[], [], [], []


    for i, particle in {3:'noise', 2:'pion', 0:'muon', 1:'electron'}.items():

        eds.append(raw_ed[raw_labels==i])
        labels.append(greek_letter_dict.get(particle))
        colors.append(color_dict.get(particle))
        weights.append(np.ones(len(raw_ed[raw_labels==i]))/len(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)]))


    fig = plt.figure(figsize=(8, 7))
    fontsize=15
    ax = plt.gca()

    plt.text(0.08, 0.9, 'CEPC AHCAL', fontsize=fontsize, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )
    plt.text(0.08, 0.85, source+' @{}GeV'.format(ep), fontsize=fontsize, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )

    # plt.text(0.08, 0.8, text_dict.get(approach), fontsize=fontsize, fontstyle='normal',
    #          horizontalalignment='left',
    #          verticalalignment='top', transform=ax.transAxes, )
    #
    # plt.text(0.1, 0.7, 'ANN threshold @{}'.format(threshold), fontsize=12, fontstyle='normal',
    #          horizontalalignment='left',
    #          verticalalignment='center', transform=ax.transAxes, )

    plt.hist(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)], bins=bins, range=[ll, ul], histtype='step',
             color='black',
             density=False, linewidth=2,alpha=0.8,log=log,
                   weights=np.ones(len(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)]))/
                           len(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)]))
    n,b=np.histogram(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)], bins=bins,range=[ll, ul],
                     weights=np.ones(len(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)]))/
                           len(raw_ed[np.logical_and(raw_labels>=0, raw_labels<=3)]))

    plt.plot(b[:-1] + 0.5 * (b[1:] - b[:-1]), n, 'o', color='black', alpha=1, label='Beam', markersize=7 )



    plt.hist(eds, bins=bins, label=labels, range=[ll, ul],
             histtype='stepfilled', color=colors, alpha=0.5,
             density=False,stacked=stacked, linewidth=3, log=log, weights=weights)

    plt.tick_params(labelsize=fontsize)

    if y_ul!=None:
        plt.ylim(0, y_ul)
    else:
        plt.ylim(0, 1.3*np.amax(n))
    plt.legend(loc='upper right', bbox_to_anchor=(0.98,0.95),fontsize=fontsize)
    plt.xlabel('Energy deposition [MeV]', fontsize=fontsize)
    plt.ylabel('# [Normalized]', fontsize=fontsize)
    plt.savefig(os.path.join(save_dir, 'e_dep_composition_{}GeV_{}.png'.format(ep, approach)))
    # plt.show()
    plt.close(fig)

if __name__ == '__main__':
    # ep_list = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120]
    ep_list = [20, 50, 80]

    for ep in  ep_list:
        for ap in ['tb']:
            plot_tot_e_purifiled_compare(
                raw_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_v4_{}/{}GeV.csv'.format(ap,ep),
                save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2023_sps/{}GeV_{}'.format(ep,ap),
                source='SPS-H2 Pion beam',
                approach=ap,
                stacked=True,
                ep=ep,
                log=False,
                bins=100

            )
    # ep = 10
    # for ap in ['mc', 'tb']:
    #     plot_tot_e_purifiled_compare(
    #         raw_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_{}/{}GeV.csv'.format(
    #             ap, ep),
    #         save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}'.format(ep, ap),
    #         source='SPS-H8 Pion beam',
    #         approach=ap,
    #         stacked=True,
    #         ep=ep,
    #         log=False,
    #         bins=30,
    #         ul=300,
    #         y_ul=0.25
    #
    #     )
    # #
    # ep = 50
    # for ap in ['mc', 'tb']:
    #     plot_tot_e_purifiled_compare(
    #         raw_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_{}/{}GeV.csv'.format(
    #             ap, ep),
    #         save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}'.format(ep, ap),
    #         source='SPS-H8 Pion beam',
    #         approach=ap,
    #         stacked=True,
    #         ep=ep,
    #         log=False,
    #         bins=100,
    #         ul=1200,
    #         y_ul=None
    #
    #     )
    #
    # ep = 80
    # for ap in ['mc', 'tb']:
    #     plot_tot_e_purifiled_compare(
    #         raw_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_{}/{}GeV.csv'.format(
    #             ap, ep),
    #         save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}'.format(ep, ap),
    #         source='SPS-H8 Pion beam',
    #         approach=ap,
    #         stacked=True,
    #         ep=ep,
    #         log=False,
    #         bins=30,
    #         ul=1800,
    #         y_ul=0.3
    #
    #     )
    #
    #
    # ep=120
    # for ap in ['mc', 'tb']:
    #     plot_tot_e_purifiled_compare(
    #         raw_file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_{}/{}GeV.csv'.format(
    #             ap, ep),
    #         save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_{}'.format(ep, ap),
    #         source='SPS-H8 Pion beam',
    #         approach=ap,
    #         stacked=True,
    #         ep=ep,
    #         log=False,
    #         bins=30,
    #         ul=2500,
    #         y_ul=0.5
    #
    #     )
    pass

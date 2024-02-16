#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/15 20:35
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : analyze_saturation.py
# @Software: PyCharm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ana(file_dict):


    ratio = []

    for ep, file_path_dict in file_dict.items():
        mc_var = pd.read_csv(file_path_dict.get('mc'))
        e_var = pd.read_csv(file_path_dict.get('e'))
        pi_var = pd.read_csv(file_path_dict.get('pi'))

        mc_label = mc_var['particle_label']
        e_label = e_var['particle_label']
        pi_label = pi_var['particle_label']

        mc_column = mc_var['E_dep']
        e_column = e_var['E_dep']
        pi_column = pi_var['E_dep']

        range = [np.amin(np.concatenate([mc_column, e_column, pi_column])),
                 np.amax(np.concatenate([mc_column, e_column, pi_column]))]

        bins = 100

        n_mc_e, b_mc_e = np.histogram(mc_column[mc_label == 1], bins=bins, range=range, )
        n_mc_pi, b_mc_pi = np.histogram(mc_column[mc_label == 2], bins=bins, range=range, )

        n_e, b_e = np.histogram(e_column[e_label == 1], bins=bins, range=range, )
        n_pi, b_pi = np.histogram(pi_column[pi_label == 2], bins=bins, range=range, )

        mc_e_peak = b_mc_e[np.argmax(n_mc_e)] + 0.5 * (b_mc_e[1] - b_mc_e[0])
        mc_pi_peak = b_mc_pi[np.argmax(n_mc_pi)] + 0.5 * (b_mc_pi[1] - b_mc_pi[0])

        e_peak = b_e[np.argmax(n_e)] + 0.5 * (b_e[1] - b_e[0])
        pi_peak = b_pi[np.argmax(n_pi)] + 0.5 * (b_pi[1] - b_pi[0])

        ratio.append([mc_e_peak, e_peak, mc_pi_peak ,pi_peak])

    ratio = np.array(ratio)

    df = pd.DataFrame(
        {'mc_e_peak': ratio[:, 0],
         'e_peak': ratio[:, 1],
         'mc_pi_peak': ratio[:, 2],
         'pi_peak': ratio[:, 3],
         'ep':file_dict.keys(),
         },
        index=file_dict.keys()
    )

    df.to_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/ratio.csv')
if __name__ == '__main__':
    file_dict = {
        10: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/10GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_10_run276_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023/10GeV_pi/bdt_var.csv'
        },
        30: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/30GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_30_run274_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_30_run250_2023/bdt_var.csv'
        },
        50: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/50GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_50_run272_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_50_run245_2023/bdt_var.csv'
        },

        60: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/60GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_60_run265_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_60_run216_2023/bdt_var.csv'
        },
        80: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/80GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_80_run266_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_80_run220_2023/bdt_var.csv'
        },
        100: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/100GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_100_run267_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_100_run230_2023/bdt_var.csv'
        },
        120: {
            'mc': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515/120GeV/bdt_var.csv',
            'e': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/e_120_run268_2023/bdt_var.csv',
            'pi': '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_120_run236_2023/bdt_var.csv'
        },

    }
    # ana()
    # ratio = pd.read_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/ratio.csv')
    #
    #
    # fig=plt.figure(figsize=(6,5))
    # x=ratio['ep']
    # plt.plot(x,ratio['mc_e_peak']/ratio['e_peak'],'o-',label= 'e', color='blue')
    # plt.plot(x, ratio['mc_pi_peak']/ratio['pi_peak'], 'o-', label='pi', color='red')
    # plt.legend()
    # plt.xticks(x)
    # plt.xlabel('E [GeV]', fontsize=13)
    # plt.ylabel(r'$E_{peak}^{mc}/E_{peak}^{data}$',fontsize=13)
    # plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(6, 5))
    #
    # plt.plot(x, ratio['e_peak'], 'o-', label='Data e', color='blue')
    # plt.plot(x, ratio['pi_peak'], 'o-', label='Data pi', color='red')
    #
    # plt.plot(x, ratio['mc_e_peak'], '*--', label='MC e', color='blue')
    # plt.plot(x, ratio['mc_pi_peak'], '*--', label='MC pi', color='red')
    # plt.legend()
    # plt.xticks(x)
    # plt.xlabel('E [GeV]', fontsize=13)
    # plt.ylabel(r'$E_{peak}$'+ ' [GeV]', fontsize=13)
    # plt.show()
    # plt.close(fig)

    ratio = pd.read_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/ratio.csv')
    e_ratio=list(ratio['mc_e_peak'].values/ratio['e_peak'].values)
    p_ratio=list(ratio['mc_pi_peak'].values/ratio['pi_peak'].values)
    ep=list(ratio['ep'].values)

    df = pd.DataFrame(
        {
            'ep': ep,
            'e_ratio': e_ratio,
            'p_ratio': p_ratio,
        }
    )
    df.to_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/ratio_0.csv', index=False)

    ep_new=np.arange(10, 121,10)
    e_ratio_new=[]
    p_ratio_new=[]

    for i,e in enumerate(ep_new):
        if e in ep:
            e_ratio_new.append(e_ratio[ep.index(e)])
            p_ratio_new.append(p_ratio[ep.index(e)])

        else:
            e_ratio_new.append(0.5*(e_ratio[ep.index(e-10)]+e_ratio[ep.index(e+10)]))
            p_ratio_new.append(0.5*(p_ratio[ep.index(e-10)]+p_ratio[ep.index(e+10)]))

    df=pd.DataFrame(
        {
            'ep':ep_new,
            'e_ratio':e_ratio_new,
            'p_ratio':p_ratio_new,
        }
    )
    df.to_csv('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/ratio_1.csv', index=False)
    pass

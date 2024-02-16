#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/18 15:51
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : calculate_energy_resolution.py
# @Software: PyCharm
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math

class Energy_Response():

    def __init__(self, file_path):

        self.df=pd.read_csv(file_path)
        self.cut=True*np.ones(len(self.df.values))

        self.popt, self.pcov=None, None

    def shower_start_cut(self, threshold):
        self.cut=np.logical_and(self.cut, self.df['Shower_start']<threshold)

    def shower_radius_cut(self, threshold):
        self.cut=np.logical_and(self.cut, self.df['Shower_radius']>threshold)

    def e_dep_cut(self, threshold):
        self.cut = np.logical_and(self.cut, self.df['E_dep'] > threshold)

    def gaussian_func(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def crystal_ball_func(self, x, alpha,  mean, sigma, n):
        A = ((n / abs(alpha)) ** n) * math.exp(-1 * ((alpha ** 2) / 2))
        B = n / abs(alpha) - abs(alpha)
        C = n / abs(alpha) / (n - 1) * math.exp(-1 * (alpha ** 2) / 2)
        D = math.sqrt(math.pi / 2) * (1 + math.erf(abs(alpha) / math.sqrt(2)))
        N = 1 / (sigma * (C + D))

        return np.where((x - mean) / sigma > (-1 * alpha), N * np.exp(-1 * (np.square(x - mean)) / (2 * sigma ** 2)),
                        N * A * np.power((B - (x - mean) / sigma), (-1 * n)))

    def get_gauss_para(self, bins, label, n_sigma):

        density=True
        e_dep=self.df['E_dep'][np.logical_and(self.cut, self.df['Particle_label']==label)]
        std = np.std(e_dep)
        mean = np.mean(e_dep)
        ll = mean - n_sigma * std
        ul = mean + n_sigma * std

        n, b = np.histogram(e_dep, bins=bins, range=[ll, ul], density=density)

        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]

        e_peak = b[np.argmax(n)]

        ll = e_peak - n_sigma * std
        ul = e_peak + n_sigma * std

        n, b = np.histogram(e_dep, bins=bins, range=[ll, ul], density=density)

        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]



        self.gauss_popt, self.gauss_pcov = curve_fit(self.gaussian_func, b, n, p0=[len(e_dep), mean, std])

        return ll, ul, n,b


    def get_cbf_para(self, bins, label, n_sigma):

        density=True
        e_dep = self.df['E_dep'][np.logical_and(self.cut, self.df['Particle_label'] == label)]


        std = np.std(e_dep)
        mean = np.mean(e_dep)
        ll = mean - n_sigma * std
        ul = mean + n_sigma * std

        n, b = np.histogram(e_dep, bins=bins, range=[ll, ul], density=density)

        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]

        e_peak = b[np.argmax(n)]

        ll = e_peak - n_sigma * std
        ul = e_peak + n_sigma * std

        n, b = np.histogram(e_dep, bins=bins, range=[ll, ul], density=density)

        b = b + np.mean(np.diff(b)) / 2
        b = b[:len(n)]

        self.cbf_popt, self.cbf_pcov = curve_fit(self.crystal_ball_func, b, n,  p0=[0.85,  b[np.argmax(n)], 100, 4.2])

        return ll, ul, n, b

    def plot_variable(self, variable, bins):

        fig=plt.Figure(figsize=(8,7))
        plt.hist(self.df[variable][self.cut],bins, histtype='stepfilled')
        plt.show()
        plt.close(fig)


    def plot_E_dep_gauss(self, label, bins,  ep, n_sigma,y_ul, save_path):

        label_text={
            1:'Electron',
            2:'Pion'
        }


        fontsize=13

        ll, ul, n, b=self.get_gauss_para(bins=bins, label=label, n_sigma=n_sigma)
        param_errors = np.sqrt(np.diag(self.gauss_pcov))

        fig = plt.figure(figsize=(8, 7))

        ax = plt.gca()

        plt.text(0.05, 0.9, 'CEPC Test Beam', fontsize=fontsize, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        plt.text(0.05, 0.8, '{} beam @{}GeV'.format(label_text.get(label), ep), fontsize=fontsize, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        # plt.hist(e_dep, bins=bins, label='data', range=[ll, ul], histtype='stepfilled', color='red', alpha=0.5,
        #          density=False)

        plt.plot(b[n>0], n[n>0], 'o', color='black', alpha=1, label='{} Data'.format(label_text.get(label)), markersize=6)

        plt.tick_params(labelsize=fontsize)

        e_mean=str(round(self.gauss_popt[1], 2))
        e_meam_err=str(round(param_errors[1],2))
        e_sigma=str(round(self.gauss_popt[2], 2))
        e_sigma_err = str(round(param_errors[2], 2))

        fit_label='E = '+e_mean+r'$\pm$'+ e_meam_err+'\n' + chr(963)+' = '+e_sigma+r'$\pm$'+ e_sigma_err
        plt.plot(b, self.gaussian_func(b, *self.gauss_popt), label=fit_label, linewidth=3, color='red')
        plt.legend(loc='upper right', bbox_to_anchor=(0.98,0.92),fontsize=fontsize-2)
        plt.ylabel('#', fontsize=fontsize)
        plt.xlabel('[MeV]', fontsize=fontsize)

        if y_ul==None:
            y_ul=1.3*np.amax(n)

        plt.ylim(0, y_ul)

        plt.savefig(save_path)
        # plt.show()
        plt.close(fig)


    def plot_E_dep_cbf(self, label, bins,  ep, n_sigma,y_ul, save_path):

        label_text={
            1:'Electron',
            2:'Pion'
        }

        density=True
        fontsize=13

        ll, ul, n, b=self.get_cbf_para(bins=bins, label=label, n_sigma=n_sigma)
        param_errors = np.sqrt(np.diag(self.cbf_pcov))

        fig = plt.figure(figsize=(8, 7))

        ax = plt.gca()

        plt.text(0.05, 0.9, 'CEPC Test Beam', fontsize=fontsize, fontstyle='oblique', fontweight='bold',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        plt.text(0.05, 0.8, '{} beam @{}GeV'.format(label_text.get(label), ep), fontsize=fontsize, fontstyle='normal',
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, )
        # plt.hist(e_dep, bins=bins, label='data', range=[ll, ul], histtype='stepfilled', color='red', alpha=0.5,
        #          density=False)

        plt.plot(b[n>0], n[n>0], 'o', color='black', alpha=1, label='{} Data'.format(label_text.get(label)), markersize=6)

        plt.tick_params(labelsize=fontsize)

        e_mean=str(round(self.cbf_popt[1], 2))
        e_meam_err=str(round(param_errors[1],2))
        e_sigma=str(round(self.cbf_popt[2], 2))
        e_sigma_err = str(round(param_errors[2], 2))

        fit_label='E = '+e_mean+r'$\pm$'+ e_meam_err+'\n' + chr(963)+' = '+e_sigma+r'$\pm$'+ e_sigma_err
        plt.plot(b, self.crystal_ball_func(b, *self.cbf_popt), label=fit_label, linewidth=3, color='red')
        plt.legend(loc='upper right', bbox_to_anchor=(0.98,0.92),fontsize=fontsize-2)
        plt.ylabel('#', fontsize=fontsize)
        plt.xlabel('[MeV]', fontsize=fontsize)

        if y_ul==None:
            y_ul=1.3*np.amax(n)

        plt.ylim(0, y_ul)

        plt.savefig(save_path)
        # plt.show()
        plt.close(fig)


def resolution_func(x, a, b):
    return np.sqrt((a / np.sqrt(x))**2 + b**2)

def plot_resolution(df, func, save_path):



    x = df['ep'].values
    y = df['resolution'].values
    y_error=df['res_err'].values

    x_base = np.linspace(x[0], x[-1], 100)
    y_base = np.sqrt(np.square(0.6 / np.sqrt(x_base)) + 0.03**2)

    popt, pcov = curve_fit(func, x, y)

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.text(0.15, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.15, 0.8, 'SPS-H2 {}- Beam'.format(chr(960)), fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )


    plt.errorbar(x, y, y_error, linestyle='',capsize=3,label='Data', color='red')
    # plt.plot(x, y, '.', label='Data', color='red')

    plt.plot(x_base, y_base, '--', label='Target ' + r'$\frac{60\%}{\sqrt{E}}\oplus3\%$', color='green')

    label='{}- Resolution '.format(chr(960)) + r'$\frac{'+str(round(popt[0] * 100, 1))+'\%}{\sqrt{E}}\oplus'\
              +str(round(popt[1] * 100,1))+'\%$'

    plt.plot(x_base, func(x_base, *popt),
             label=label, color='black')

    plt.legend(loc='center right')

    plt.xticks(x)
    plt.xlabel('E [GeV]')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_linearity(x,y, save_path, ep_list):

    x_base = np.linspace(ep_list[0], ep_list[-1], 10)

    func= lambda x, a, b: a*x+b

    popt, pcov = curve_fit(func, x, y)

    fig, axs = plt.subplots(2, 1, sharex='none', tight_layout=True, gridspec_kw={'height_ratios': [5, 2]},
                            figsize=(6, 8))

    axs = axs.flatten()

    axs[0].text(0.15, 0.9, 'CEPC Test Beam', fontsize=18, fontstyle='oblique', fontweight='bold',
                horizontalalignment='left',
                verticalalignment='center', transform=axs[0].transAxes, )
    axs[0].text(0.15, 0.8, 'Energy linearity', fontsize=15, fontstyle='normal',
                horizontalalignment='left',
                verticalalignment='center', transform=axs[0].transAxes, )

    axs[0].plot(x, y, '*', label='data', markersize=10, color='red')
    axs[0].plot(x_base, func(x_base, *popt),
                label=('Fit'), color='black')

    axs[0].set_xticks(ep_list)
    axs[0].set_xlabel('E [GeV]')
    axs[0].set_ylabel('Reconstructed E [MeV]')
    axs[0].legend(loc='right')

    axs[1].plot(x, (func(np.array(x), *popt) - np.array(y)) / np.array(y)*100,'*-', markersize=10,color='black')
    axs[1].plot(x_base, 1.5 * np.ones(len(x_base)), '--', color='grey')
    axs[1].plot(x_base, -1.5 * np.ones(len(x_base)), '--', color='grey')
    axs[1].set_xticks(ep_list)
    axs[1].set_yticks(np.linspace(-3, 3, 13))
    axs[1].set_xlabel('E [GeV]')
    axs[1].set_ylabel('Linearity [%]')

    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def main_e_dep():

    # ep_list=np.linspace(10, 80, 8).astype(np.int32)
    #
    # e_mean_gauss = []
    # e_mean_err_gauss = []
    # e_sigma_gauss = []
    # e_sigma_err_gauss = []
    #
    # e_mean_cbf = []
    # e_mean_err_cbf = []
    # e_sigma_cbf = []
    # e_sigma_err_cbf = []
    #
    # for ep in ep_list:
    #     er = Energy_Response(
    #         file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_tb/{}GeV.csv'.format(ep))
    #
    #     er.shower_start_cut(threshold=4)
    #     er.shower_radius_cut(threshold=2)
    #     er.e_dep_cut(threshold=100)
    #     er.plot_E_dep_cbf(label=2, bins=100, ep=ep, n_sigma=2, y_ul=None,
    #                         save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_tb/e_dep_cbf.png'.format(
    #                             ep))
    #     er.plot_E_dep_gauss(label=2, bins=100, ep=ep, n_sigma=2, y_ul=None,
    #                           save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_tb/e_dep_gauss.png'.format(ep))
    #
    #     e_mean_gauss.append(er.gauss_popt[1])
    #     e_mean_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[1])
    #     e_sigma_gauss.append(er.gauss_popt[2])
    #     e_sigma_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[2])
    #
    #     e_mean_cbf.append(er.cbf_popt[1])
    #     e_mean_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[1])
    #     e_sigma_cbf.append(er.cbf_popt[2])
    #     e_sigma_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[2])
    #
    # resolution_gauss = np.array(e_sigma_gauss)/np.array(e_mean_gauss)
    # res_err_gauss = resolution_gauss*np.sqrt(np.square(np.array(e_sigma_err_gauss)/np.array(e_sigma_gauss))
    #                              +np.square(np.array(e_mean_err_gauss)/np.array(e_sigma_gauss)))
    #
    # df_gauss=pd.DataFrame(
    #     {'ep':ep_list,
    #      'resolution':resolution_gauss,
    #      'res_err':res_err_gauss,
    #      }
    # )
    #
    # plot_resolution(df_gauss, resolution_func, save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/performance/resolution_gauss.png')
    #
    # resolution_cbf = np.array(e_sigma_cbf) / np.array(e_mean_cbf)
    # res_err_cbf = resolution_cbf * np.sqrt(np.square(np.array(e_sigma_err_cbf) / np.array(e_sigma_cbf))
    #                                            + np.square(np.array(e_mean_err_cbf) / np.array(e_sigma_cbf)))
    #
    # df_cbf = pd.DataFrame(
    #     {'ep': ep_list,
    #      'resolution': resolution_cbf,
    #      'res_err': res_err_cbf,
    #      }
    # )
    #
    # plot_resolution(df_cbf, resolution_func,
    #                 save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/performance/resolution_cbf.png')


    ep_list = np.linspace(10, 80, 8).astype(np.int32)

    e_mean_gauss = []
    e_mean_err_gauss = []
    e_sigma_gauss = []
    e_sigma_err_gauss = []

    e_mean_cbf = []
    e_mean_err_cbf = []
    e_sigma_cbf = []
    e_sigma_err_cbf = []

    for ep in ep_list:
        er = Energy_Response(
            file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_v4_tb/{}GeV.csv'.format(
                ep))

        er.shower_start_cut(threshold=4)
        er.shower_radius_cut(threshold=2)

        if ep==10:
            er.shower_radius_cut(threshold=2)
        else:
            er.shower_radius_cut(threshold=2)

        er.e_dep_cut(threshold=100)


        n_signa=6

        er.plot_E_dep_cbf(label=2, bins=100, ep=ep, n_sigma=n_signa, y_ul=None,
                            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2023_sps/{}GeV_tb/e_dep_cbf.png'.format(
                                ep))

        er.plot_E_dep_gauss(label=2, bins=100, ep=ep, n_sigma=2, y_ul=None,
                            save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2023_sps/{}GeV_tb/e_dep_gauss.png'.format(
                                ep))

        e_mean_gauss.append(er.gauss_popt[1])
        e_mean_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[1])
        e_sigma_gauss.append(er.gauss_popt[2])
        e_sigma_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[2])

        e_mean_cbf.append(er.cbf_popt[1])
        e_mean_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[1])
        e_sigma_cbf.append(er.cbf_popt[2])
        e_sigma_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[2])

    resolution_gauss = np.array(e_sigma_gauss) / np.array(e_mean_gauss)
    res_err_gauss = resolution_gauss * np.sqrt(np.square(np.array(e_sigma_err_gauss) / np.array(e_sigma_gauss))
                                               + np.square(np.array(e_mean_err_gauss) / np.array(e_sigma_gauss)))

    df_gauss = pd.DataFrame(
        {'ep': ep_list,
         'resolution': resolution_gauss,
         'res_err': res_err_gauss,
         }
    )

    plot_resolution(df_gauss, resolution_func,
                    save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2023_sps/performance/resolution_gauss.png')

    resolution_cbf = np.array(e_sigma_cbf) / np.array(e_mean_cbf)
    res_err_cbf = resolution_cbf * np.sqrt(np.square(np.array(e_sigma_err_cbf) / np.array(e_sigma_cbf))
                                           + np.square(np.array(e_mean_err_cbf) / np.array(e_sigma_cbf)))

    df_cbf = pd.DataFrame(
        {'ep': ep_list,
         'resolution': resolution_cbf,
         'res_err': res_err_cbf,
         }
    )

    plot_resolution(df_cbf, resolution_func,
                    save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2023_sps/performance/resolution_cbf.png')

    y=np.array(e_mean_cbf)
    y[0]=151
    plot_linearity(x=ep_list,
                   y=y,
                   save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2023_sps/performance/linearity_cbf.png',
                   ep_list=ep_list)


def main_mc_e_dep():

    # ep_list=np.linspace(10, 80, 8).astype(np.int32)
    #
    # e_mean_gauss = []
    # e_mean_err_gauss = []
    # e_sigma_gauss = []
    # e_sigma_err_gauss = []
    #
    # e_mean_cbf = []
    # e_mean_err_cbf = []
    # e_sigma_cbf = []
    # e_sigma_err_cbf = []
    #
    # for ep in ep_list:
    #     er = Energy_Response(
    #         file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_tb/{}GeV.csv'.format(ep))
    #
    #     er.shower_start_cut(threshold=4)
    #     er.shower_radius_cut(threshold=2)
    #     er.e_dep_cut(threshold=100)
    #     er.plot_E_dep_cbf(label=2, bins=100, ep=ep, n_sigma=2, y_ul=None,
    #                         save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_tb/e_dep_cbf.png'.format(
    #                             ep))
    #     er.plot_E_dep_gauss(label=2, bins=100, ep=ep, n_sigma=2, y_ul=None,
    #                           save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/{}GeV_tb/e_dep_gauss.png'.format(ep))
    #
    #     e_mean_gauss.append(er.gauss_popt[1])
    #     e_mean_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[1])
    #     e_sigma_gauss.append(er.gauss_popt[2])
    #     e_sigma_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[2])
    #
    #     e_mean_cbf.append(er.cbf_popt[1])
    #     e_mean_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[1])
    #     e_sigma_cbf.append(er.cbf_popt[2])
    #     e_sigma_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[2])
    #
    # resolution_gauss = np.array(e_sigma_gauss)/np.array(e_mean_gauss)
    # res_err_gauss = resolution_gauss*np.sqrt(np.square(np.array(e_sigma_err_gauss)/np.array(e_sigma_gauss))
    #                              +np.square(np.array(e_mean_err_gauss)/np.array(e_sigma_gauss)))
    #
    # df_gauss=pd.DataFrame(
    #     {'ep':ep_list,
    #      'resolution':resolution_gauss,
    #      'res_err':res_err_gauss,
    #      }
    # )
    #
    # plot_resolution(df_gauss, resolution_func, save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/performance/resolution_gauss.png')
    #
    # resolution_cbf = np.array(e_sigma_cbf) / np.array(e_mean_cbf)
    # res_err_cbf = resolution_cbf * np.sqrt(np.square(np.array(e_sigma_err_cbf) / np.array(e_sigma_cbf))
    #                                            + np.square(np.array(e_mean_err_cbf) / np.array(e_sigma_cbf)))
    #
    # df_cbf = pd.DataFrame(
    #     {'ep': ep_list,
    #      'resolution': resolution_cbf,
    #      'res_err': res_err_cbf,
    #      }
    # )
    #
    # plot_resolution(df_cbf, resolution_func,
    #                 save_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/ANN_pid/2022_sps/performance/resolution_cbf.png')


    ep_list = np.linspace(10, 80, 8).astype(np.int32)

    e_mean_gauss = []
    e_mean_err_gauss = []
    e_sigma_gauss = []
    e_sigma_err_gauss = []

    e_mean_cbf = []
    e_mean_err_cbf = []
    e_sigma_cbf = []
    e_sigma_err_cbf = []

    for ep in ep_list:
        er = Energy_Response(
            file_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/1031_e_pi/{}GeV/bdt_var.csv'.format(
                ep))

        er.shower_start_cut(threshold=4)
        er.shower_radius_cut(threshold=2)

        if ep == 10:
            er.shower_radius_cut(threshold=2)
        else:
            er.shower_radius_cut(threshold=2)

        er.e_dep_cut(threshold=100)

        n_signa = 6

        save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/mc_energy_response/1031_{}GeV'.format(ep)
        os.makedirs(save_dir, exist_ok=True)

        er.plot_E_dep_cbf(label=2, bins=100, ep=ep, n_sigma=n_signa, y_ul=None,
                            save_path=os.path.join(save_dir, 'e_dep_cbf.png'))

        er.plot_E_dep_gauss(label=2, bins=100, ep=ep, n_sigma=2, y_ul=None,
                            save_path=os.path.join(save_dir, 'e_dep_gauss.png'))

        e_mean_gauss.append(er.gauss_popt[1])
        e_mean_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[1])
        e_sigma_gauss.append(er.gauss_popt[2])
        e_sigma_err_gauss.append(np.sqrt(np.diag(er.gauss_pcov))[2])

        e_mean_cbf.append(er.cbf_popt[1])
        e_mean_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[1])
        e_sigma_cbf.append(er.cbf_popt[2])
        e_sigma_err_cbf.append(np.sqrt(np.diag(er.cbf_pcov))[2])

    resolution_gauss = np.array(e_sigma_gauss) / np.array(e_mean_gauss)
    res_err_gauss = resolution_gauss * np.sqrt(np.square(np.array(e_sigma_err_gauss) / np.array(e_sigma_gauss))
                                               + np.square(np.array(e_mean_err_gauss) / np.array(e_sigma_gauss)))

    df_gauss = pd.DataFrame(
        {'ep': ep_list,
         'resolution': resolution_gauss,
         'res_err': res_err_gauss,
         }
    )

    save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Fig/mc_energy_response/0515'
    os.makedirs(save_dir, exist_ok=True)

    plot_resolution(df_gauss, resolution_func,
                    save_path=os.path.join(save_dir, 'resolution_gauss.png'))

    resolution_cbf = np.array(e_sigma_cbf) / np.array(e_mean_cbf)
    res_err_cbf = resolution_cbf * np.sqrt(np.square(np.array(e_sigma_err_cbf) / np.array(e_sigma_cbf))
                                           + np.square(np.array(e_mean_err_cbf) / np.array(e_sigma_cbf)))

    df_cbf = pd.DataFrame(
        {'ep': ep_list,
         'resolution': resolution_cbf,
         'res_err': res_err_cbf,
         }
    )

    plot_resolution(df_cbf, resolution_func,
                    save_path=os.path.join(save_dir, 'resolution_cdf.png'))

    y=np.array(e_mean_cbf)
    plot_linearity(x=ep_list,
                   y=y,
                   save_path=os.path.join(save_dir, 'linearity_cbf.png'),
                   ep_list=ep_list)
if __name__ == '__main__':

    main_e_dep()
    main_mc_e_dep()
    pass

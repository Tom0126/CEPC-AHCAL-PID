#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 23:02
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : main.py
# @Software: PyCharm
import os

import matplotlib.pyplot as plt

from variable import ShowerVariable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imgs_file_path", type=str, help="npy file.")
parser.add_argument("--mu_path", type=str, help="npy file.")
parser.add_argument("--e_path", type=str, help="npy file.")
parser.add_argument("--pi_path", type=str, help="npy file.")
parser.add_argument("--labels_file_path", type=str, help="npy file.")
parser.add_argument("--save_dir", type=str, help="dir for save.")
args = parser.parse_args()


def main(mu_path,e_path,pi_path,save_dir, layer_nums=40, granularity=18):
    mu_data=ShowerVariable(mu_path)
    mu_layers_fired, mu_cells_fired, mu_max_cells_fired=mu_data.fired()
    mu_hit_layer_ratio = mu_cells_fired/mu_layers_fired

    e_data = ShowerVariable(e_path)
    e_layers_fired, e_cells_fired, e_max_cells_fired = e_data.fired()
    e_hit_layer_ratio = e_cells_fired / e_layers_fired

    pi_data = ShowerVariable(pi_path)
    pi_layers_fired, pi_cells_fired, pi_max_cells_fired = pi_data.fired()
    pi_hit_layer_ratio = pi_cells_fired / pi_layers_fired



    # layers fired
    fig=plt.figure(figsize=(6,5))
    ax=plt.gca()
    plt.hist(mu_layers_fired,bins=int(layer_nums/4),histtype='step', linewidth=1.5, label='mu+',density=True)
    plt.hist(e_layers_fired, bins=int(layer_nums/4), histtype='step', linewidth=1.5, label='e+',density=True)
    plt.hist(pi_layers_fired, bins=int(layer_nums/4), histtype='step', linewidth=1.5, label='pi+',density=True)
    plt.xlim(0,layer_nums+5)
    plt.text(0.1, 0.9, 'AHCAL Simulation',fontsize=15, fontstyle='oblique', fontweight='bold',transform=ax.transAxes,horizontalalignment='left',)
    plt.text(0.1, 0.85, 'Layers Fired', fontsize=10, transform=ax.transAxes,horizontalalignment='left',)
    plt.legend(loc='upper left',  bbox_to_anchor=(0.1, 0.8),bbox_transform=ax.transAxes)
    plt.savefig(os.path.join(save_dir,'layers_fired.png'))
    plt.close(fig)

    # Average hits per layer fired
    fig= plt.figure(figsize=(6, 5))
    ax = plt.gca()
    plt.hist(mu_hit_layer_ratio, bins=layer_nums, histtype='step', linewidth=1.5, label='mu+', density=True)
    plt.hist(e_hit_layer_ratio, bins=layer_nums, histtype='step', linewidth=1.5, label='e+', density=True)
    plt.hist(pi_hit_layer_ratio, bins=layer_nums, histtype='step', linewidth=1.5, label='pi+', density=True)
    plt.text(0.1, 0.9, 'AHCAL Simulation', fontsize=15, fontstyle='oblique', fontweight='bold', transform=ax.transAxes,
             horizontalalignment='left', )
    plt.text(0.1, 0.85, 'Average Hits Number in Layer', fontsize=10, transform=ax.transAxes, horizontalalignment='left', )
    plt.legend(loc='upper left',  bbox_to_anchor=(0.1, 0.8),bbox_transform=ax.transAxes)
    plt.savefig(os.path.join(save_dir, 'average_layers_fired.png'))
    plt.close(fig)

    # Max hits per layer fired
    fig= plt.figure(figsize=(6, 5))
    ax = plt.gca()
    plt.hist(mu_max_cells_fired, bins=granularity, histtype='step', linewidth=1.5, label='mu+', density=True)
    plt.hist(e_max_cells_fired, bins=granularity, histtype='step', linewidth=1.5, label='e+', density=True)
    plt.hist(pi_max_cells_fired, bins=granularity, histtype='step', linewidth=1.5, label='pi+', density=True)
    plt.text(0.1, 0.9, 'AHCAL Simulation', fontsize=15, fontstyle='oblique', fontweight='bold', transform=ax.transAxes,
             horizontalalignment='left', )
    plt.text(0.1, 0.85, 'Maximum Hits Number in Layer', fontsize=10, transform=ax.transAxes, horizontalalignment='left', )
    plt.legend(loc='upper left',  bbox_to_anchor=(0.1, 0.8),bbox_transform=ax.transAxes)
    plt.xlim(0, layer_nums)
    plt.savefig(os.path.join(save_dir, 'maximum_hits_fired.png'))
    plt.close(fig)


if __name__ == '__main__':
    main(args.mu_path, args.e_path, args.pi_path, args.save_dir)
    pass

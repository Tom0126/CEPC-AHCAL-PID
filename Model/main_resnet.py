#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 21:18
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : Interface.py
# @Software: PyCharm


# -*- coding: utf-8 -*-
"""
# @file name  : Train.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 15:09:00
# @brief      : CEPC PID
"""
import os
from Net.resnet import Bottleneck, BasicBlock, ResNet, ResNet_Avg
from Config.config import parser
from Data import loader
import sys
from PID import pid_data_loader_gnn, pid_data_loader
from Interface import interface
import numpy as np

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = parser.parse_args()

    net_name = '0120_{}_mc_resnet_avg_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}_st_{}_fk_{}_fs_{}_fp_{}_v1'.format(
        args.index,
        args.n_epoch,
        args.learning_rate,
        args.batch_size,
        args.optim,
        args.n_classes,
        args.l_gamma,
        args.step,
        args.short_cut,
        args.f_k,
        args.f_s,
        args.f_p,
    )



    data_dir_dict = {
        2: '../data/ihep_mc/dummy_data',
    }
    eval_datasets_dir_dict = {
        2: data_dir_dict.get(args.n_classes) + '/Validation',
    }
    data_dir_format = os.path.join(data_dir_dict.get(args.n_classes), 'Test')

    net_name = 'debug_' + net_name




    ckp_dir = os.path.join('../CheckPoint', net_name)

    interface(hyper_para={'net_used': 'resnet',
                          'n_classes': args.n_classes,
                          'batch_size': args.batch_size,
                          'n_epoch': args.n_epoch,
                          'l_r': args.learning_rate,
                          'step':args.step,
                          'l_gamma': args.l_gamma,
                          'optim': args.optim,
                          'start_planes':40,
                          'layers': {'Res18': [2,2,2,2]},
                          'block': {'Bottleneck': Bottleneck},
                          'short_cut': args.short_cut,
                          'first_kernal': args.f_k,
                          'first_stride': args.f_s,
                          'first_padding':args.f_k,
                          'val_interval':args.val_interval,
                          'scheduler': args.lr_schedule,
                          },
              net=ResNet_Avg,
              data_loader_func=loader.data_loader,
              data_set_dir=data_dir_dict.get(args.n_classes),
              ckp_dir=ckp_dir,
              eval_para={'root_path': ckp_dir,
                         'n_classes': args.n_classes,
                         'data_loader_func': loader.data_loader,
                         'combin_datasets_dir_dict': eval_datasets_dir_dict,
                         'sep_datasets_dir_dict': None,
                         'data_type': 'mc',
                         'fig_dir_name': 'Fig',
                         'threshold': 0,
                         'threshold_num': 21,
                         },

              ann_eval_para={'ckp_dir': ckp_dir,
                             'data_dir_format': data_dir_format,
                             'n_classes': args.n_classes,
                             'pid_data_loader_func': pid_data_loader,
                             'ann_signal_label_list': [0, 1],
                             'effi_points': np.linspace(1, 0.95, 111),
                             'max_nodes': -1,
                             },
              TRAIN=bool(args.train),
              EVAL=bool(args.eval),
              ANN_EVAL=bool(args.ana_eval)

              )

    pass

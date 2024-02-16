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
from Net.grav_net import GravNetModule_v2, GlobalExchangeMeanMax, GravNet
from Config.config import parser
from Data import loader
import sys
from PID import pid_data_loader_gnn
from Interface import interface

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = parser.parse_args()

    net_name = '0122_{}_mc_gravnet_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_nodes_{}_k_{}_v1'.format(
        args.index,
        args.n_epoch,
        args.learning_rate,
        args.batch_size,
        args.optim,
        args.n_classes,
        args.max_nodes,
        args.k
    )



    data_dir_dict = {
        2: '../data/ihep_mc/tutorial',
    }
    eval_datasets_dir_dict = {
        2: data_dir_dict.get(args.n_classes) + '/Validation',
    }
    data_dir_format = os.path.join(data_dir_dict.get(args.n_classes), 'Test')
    net_name = 'debug_' + net_name




    ckp_dir = os.path.join('../CheckPoint', net_name)

    interface(hyper_para={'net_used': 'gravnet_v2',
                          'n_classes': args.n_classes,
                          'batch_size': args.batch_size,
                          'n_epoch': args.n_epoch,
                          'l_r': args.learning_rate,
                          'optim': args.optim,
                          'scheduler': 'cos',

                          'max_nodes': args.max_nodes,
                          'k': args.k,
                          'in_channel': 4,
                          'in_channel_factor':3,
                          'grav_inner_channel':64,
                          'space_dimensions': 4,
                          'propagate_dimensions':64,
                          'gravnet_module':GravNetModule_v2,
                          'global_exchange_block':GlobalExchangeMeanMax,
                          },
              net=GravNet,
              data_loader_func=loader.data_loader_gnn,
              data_set_dir=data_dir_dict.get(args.n_classes),
              ckp_dir=ckp_dir,
              eval_para={'root_path': ckp_dir,
                         'n_classes': args.n_classes,
                         'data_loader_func': loader.data_loader_gnn,
                         'combin_datasets_dir_dict': eval_datasets_dir_dict,
                         'sep_datasets_dir_dict': None,
                         'data_type': 'mc',
                         'fig_dir_name': 'Fig',
                         'threshold': 0,
                         'threshold_num': 21,
                         'max_nodes': args.max_nodes
                         },

              ann_eval_para={'ckp_dir': ckp_dir,
                             'data_dir_format': data_dir_format,
                             'n_classes': args.n_classes,
                             'pid_data_loader_func': pid_data_loader_gnn,
                             'max_nodes': args.max_nodes,
                             'ann_signal_label_list': [0, 1],
                             'effi_points': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995][::-1],
                             },



              )

    pass

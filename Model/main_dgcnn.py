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
from Net.dgcnn import DGCNN_cls, get_graph_feature
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

    net_name = '0119_{}_mc_dgcnn_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}_nodes_{}_k_{}_v1'.format(
        args.index,
        args.n_epoch,
        args.learning_rate,
        args.batch_size,
        args.optim,
        args.n_classes,
        args.l_gamma,
        args.step,
        args.max_nodes,
        args.k
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

    interface(hyper_para={'net_used': 'dgcnn',
                          'n_classes': args.n_classes,
                          'batch_size': args.batch_size,
                          'n_epoch': args.n_epoch,
                          'l_r': args.learning_rate,
                          'optim': args.optim,
                          'max_nodes': args.max_nodes,
                          'k': args.k,
                          'in_channel': 4,
                          'channels': {'paper': [64, 64, 128, 256]},
                          'kernels': {'paper': [1, 1, 1, 1]},
                          'bns': {'paper': [True, True, True, True]},
                          'acti': {'paper': [True, True, True, True]},
                          'get_f_func': {'paper': get_graph_feature},
                          'adaptive_pool': None,
                          'pool_out_size': None,
                          'emb_dims': 1024,
                          'dropout': 0.5,
                          'scheduler':'cos'
                          },
              net=DGCNN_cls,
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
                             'effi_points': [0.95, 0.96, 0.97, 0.98, 0.99][::-1],
                             },

              TRAIN=True,
              ck_ann_info=True

              )

    pass

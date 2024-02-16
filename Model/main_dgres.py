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
from Net.dgcnn import DGCNN_cls, get_graph_feature, PCNN
from Net.resnet import ResNet, ResNet_Avg, Bottleneck, BasicBlock
from Net.dgres import DGRes
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

    net_name = '0121_{}_mc_dgres_epoch_{}_lr_{}_batch_{}_optim_{}_classes_{}_l_gamma_{}_step_{}_nodes_{}_k_{}_fk_{}' \
               '_fs_{}_fp_{}_v1'.format(
        args.index,
        args.n_epoch,
        args.learning_rate,
        args.batch_size,
        args.optim,
        args.n_classes,
        args.l_gamma,
        args.step,
        args.max_nodes,
        args.k,
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

    net_name='debug_'+net_name



    ckp_dir = os.path.join('../CheckPoint', net_name)

    interface(hyper_para={'net_used': 'dgres',
                          'n_classes': args.n_classes,
                          'batch_size': args.batch_size,
                          'n_epoch': args.n_epoch,
                          'l_r': args.learning_rate,
                          'scheduler': args.lr_schedule,
                          'step': args.step,
                          'l_gamma': args.l_gamma,
                          'optim': args.optim,
                          'max_nodes': args.max_nodes,
                          'PCNN_block':{'PCNN':PCNN},
                          'k': args.k,
                          'in_channel': 4,
                          'channels': {'paper': [64, 64, 128, 256, 40]},
                          'kernels': {'paper': [1, 1, 1, 1, 1]},
                          'bns': {'paper': [True, True, True, True, False]},
                          'acti': {'paper': [True, True, True, True, False]},
                          'get_f_func': {'paper': get_graph_feature},
                          'adaptive_pool': True,
                          'pool_out_size': {'18_18':(18,18)},
                          'Res_block':{'res_avg':ResNet_Avg},
                          'start_planes': 40,
                          'layers': {'Res18': [2, 2, 2, 2]},
                          'block': {'Bottleneck': Bottleneck},
                          'short_cut': args.short_cut,
                          'first_kernal': args.f_k,
                          'first_stride': args.f_s,
                          'first_padding': args.f_k,


                          },
              net=DGRes,
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

              TRAIN=True,
              ck_ann_info=True

              )

    pass

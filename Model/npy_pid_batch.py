#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 16:35
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : pid_batch.py
# @Software: PyCharm

from PID import npyPID
import glob
import os
import argparse
from Net.lenet import LeNet_bn
from Net.resnet import ResNet,BasicBlock,Bottleneck, ResNet_Avg


def npy_pid(file_dir,save_dir, model_path, n_classes, net_used, net_dict, net_para_dict,):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for file_path in glob.glob(os.path.join(file_dir,'imgs.npy')):
        # file_path=os.path.join(file_dir,file_name)

        save_path=file_path.replace(file_dir,save_dir)
        save_path=save_path.replace('.npy','_ANN.root')

        npyPID(file_path=file_path,save_path=save_path,model_path=model_path,n_classes=n_classes,net_used=net_used,net_dict=net_dict,net_para_dict=net_para_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--file_dir", type=str, help="file dir to PID.")
    parser.add_argument("--save_dir", type=str, help="PID save dir.")
    parser.add_argument("--model_path", type=str, help="ANN pid model path.")
    parser.add_argument("--n_classes", type=int, default=4, help="set n classes.")

    args = parser.parse_args()



    parameters = {
        'file_dir': args.file_dir,
        'save_dir': args.save_dir,
        'model_path': args.model_path,
        'n_classes': args.n_classes,

        'net_used':'resnet_avg',

        'net_dict':
            {'lenet': LeNet_bn,
            'resnet':ResNet,
             'resnet_avg': ResNet_Avg,
             },
        'net_para_dict': {
        'lenet':
            {'classes':args.n_classes
             },
        'resnet':
            {'block':Bottleneck,
               'layers': [2,2,2,2],
               'num_classes':args.n_classes,
               'start_planes':40,
             },
        'resnet_avg':
            {'block':Bottleneck,
             'layers': [2,2,2,2],
             'num_classes':args.n_classes,
             'start_planes':40,
             'first_kernal': 7,
             'first_stride': 2,
             'first_padding': 3,
             },

               }
    }

    npy_pid(**parameters)
    
    pass

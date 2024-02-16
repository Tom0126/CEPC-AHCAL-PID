#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 16:35
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : pid_batch.py
# @Software: PyCharm

from PID import PID, npyPID
import glob
import os
import argparse
from Net.lenet import LeNet_bn
from Net.resnet import ResNet,BasicBlock,Bottleneck
def pid(file_dir,save_dir, model_path, n_classes, net_used, net_dict, net_para_dict, z_gap, hit_threshold):

    for file_path in glob.glob(os.path.join(file_dir,'*.root')):
        # file_path=os.path.join(file_dir,file_name)

        save_path=file_path.replace(file_dir,save_dir)
        save_path=save_path.replace('.root','_ANN.root')


        PID(file_path=file_path,save_path=save_path,model_path=model_path,n_classes=n_classes,net_used=net_used,net_dict=net_dict,net_para_dict=net_para_dict, z_gap=z_gap, hit_threshold=hit_threshold)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--file_dir", type=str, help="file dir to PID.")
    parser.add_argument("--save_dir", type=str, help="PID save dir.")
    parser.add_argument("--model_path", type=str, help="ANN pid model path.")
    parser.add_argument("--n_classes", type=str, default=4, help="set n classes.")
    parser.add_argument("--z_gap", type=int, default=30)
    parser.add_argument("--hit_threshold", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        'file_dir': args.file_dir,
        'save_dir': args.save_dir,
        'model_path': args.model_path,
        'n_classes': args.n_classes,
        'z_gap': args.z_gap,
        'hit_threshold': args.hit_threshold,
        'net_used':'resnet',
        'net_dict':
            {'lenet': LeNet_bn,
            'resnet':ResNet
             },
        'net_para_dict': {
        'lenet':
            {'classes':args.n_classes
             },
        'resnet':
            {'block':Bottleneck,
               'layers': [2,2,2,2],
               'num_classes':4,
               'start_planes':40,
             }
               }
    }

    pid(**parameters)



    
    pass

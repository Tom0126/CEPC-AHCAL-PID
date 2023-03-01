#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 22:12
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : IdentifyParticles.py
# @Software: PyCharm


import torch
import numpy as np
from Net import lenet
from Config.config import parser
import matplotlib.pyplot as plt
from Data import loader
import os
from torch.nn import Softmax
# from Train import ckp_dir, MEAN, STD
from ANA.acc import plotACC
from ANA.acc_extra import plotACCExtra
from ANA.distribution import plotDistribution
from ANA.roc import plotROC
from Config.config import parser
from torchmetrics.classification import MulticlassROC, MulticlassAUROC
import uproot
from Data import ReadRoot

def readRootFileCell(path):

    file = uproot.open(path)
    # get keys
    # print(file.keys())

    # A TTree File
    simulation = file['T']

    # Get data->numpy
    data = simulation.arrays(library="np")

    # e+,x,y,z
    # hcal_energy = data['Hit_Energy']
    # x = data['Hit_X']
    # y = data['Hit_Y']
    # z = data['Hit_Z']
    hcal_energy = data['hcal_energy']
    x = data['hcal_x']
    y = data['hcal_y']
    z = data['hcal_z']

    return hcal_energy, x, y, z

def makeDatasets(file_path):
    '''
    1: inout: root file
    2: output: numpy array NCHW (,40,18,18)
    '''
    # read raw root file
    hcal_energy, x, y, z = readRootFileCell(file_path)
    num_events = len(hcal_energy)
    assert num_events == len(x)
    assert num_events == len(y)
    assert num_events == len(z)
    # NHWC
    depoits = np.zeros((num_events, 18, 18, 40))
    for i in range(num_events):
        energies_ = hcal_energy[i]
        x_ = ((x[i] + 340) / 40).astype(int)
        y_ = ((y[i] + 340) / 40).astype(int)
        z_ = ((z[i] - 301.5) / 25).astype(int)
        num_events_ = len(energies_)
        assert num_events_ == len(x_)
        assert num_events_ == len(y_)
        assert num_events_ == len(z_)
        for j in range(num_events_):
            depoits[i, x_[j], y_[j], z_[j]] += energies_[j]
    # NCHW
    # depoits = np.transpose(depoits, (0, 3, 1, 2))
    # np.save(save_path, depoits)
    return depoits

def PID(file_path, save_path, model_path, particle_type, threshold=0, n_classes=3):
    '''
    mu+: 0,
    e+: 1,
    pi+: 2,
    :param file_path:
    :param save_path:
    :param n_classes:
    :return:
    '''
    gpu = torch.cuda.is_available()

    device = 'cuda' if gpu else 'cpu'

    net = lenet.LeNet_bn(classes=n_classes)

    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    deposits = makeDatasets(file_path)
    events_number = len(deposits)

    pid_loader=loader.pid_data_loader(deposits)


    tags=None
    with torch.no_grad():
        net.eval()

        for i, inputs in enumerate(pid_loader):
            inputs=inputs.to(device)
            outputs=net(inputs)

            prbs = Softmax(dim=1)(outputs)

            prbs,particles= torch.max(prbs,1)

            if gpu:
                prbs=prbs.cpu().numpy()
                particles=particles.cpu().numpy()
            else:
                prbs = prbs.numpy()
                particles = particles.numpy()

            threshold_tag = prbs > threshold
            particle_tag = particles == particle_type

            _=np.logical_and(threshold_tag,particle_tag)

            if i==0:
                tags= _
            else:
                tags= np.concatenate((tags,_),axis=0)

    indexes=np.where(tags)[0]

    np.savetxt(save_path,indexes)






if __name__ == '__main__':
    pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 22:12
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : PID.py
# @Software: PyCharm


import torch
import numpy as np
from Net import lenet
from Data import loader
from torch.nn import Softmax
import uproot



def readRootFileCell(path):
    file = uproot.open(path)
    # get keys
    # print(file.keys())

    # A TTree File
    simulation = file['Calib_Hit']

    # Get data->numpy
    data = simulation.arrays(library="np")

    # e+,x,y,z
    hcal_energy = data['Hit_Energy']
    x = data['Hit_X']
    y = data['Hit_Y']
    z = data['Hit_Z']

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
        x_ = np.around((x[i] + 342.5491) / 40.29964).astype(int)
        y_ = np.around((y[i] + 343.05494) / 40.29964).astype(int)
        z_ = ((z[i]) / 26).astype(int)
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


def array2tree(array, save_path):

    array=np.transpose(array)
    file = uproot.recreate(save_path)
    file['Calib_Hit'] = {'ANN_mu_plus': array[0],
                         'ANN_e_plus': array[1],
                         'ANN_pi_plus': array[2],
                         }


def readPIDIndex(file_path, threshold=0.9):
    file = uproot.open(file_path)
    tree = file['Calib_Hit']

    tree = tree.arrays(library='numpy')
    mu_prbs = tree['ANN_mu_plus'].reshape(1,-1)
    positron_prbs = tree['ANN_e_plus'].reshape(1,-1)
    pi_prbs=tree['ANN_pi_plus'].reshape(1,-1)


    _=np.concatenate((mu_prbs,positron_prbs),axis=0)
    prbs=np.concatenate((_,pi_prbs),axis=0)
    prbs=np.transpose(prbs) # 0:mu+, 1: e+, 2: pi+.\


    prbs_max=np.amax(prbs,axis=1)

    particles=np.argmax(prbs,axis=1)

    tags=prbs_max> threshold

    num=len(prbs)

    assert num == len(particles)

    pid_results=np.where(tags,particles,-1*np.ones(num))

    return pid_results


def PID(file_path, save_path, model_path, n_classes=3):
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

    pid_loader = loader.pid_data_loader(deposits)

    pid_prbs = None

    with torch.no_grad():
        net.eval()

        for i, inputs in enumerate(pid_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)

            prbs = Softmax(dim=1)(outputs)

            if gpu:
                prbs = prbs.cpu().numpy()

            else:
                prbs = prbs.numpy()

            if i == 0:
                pid_prbs = prbs
            else:
                pid_prbs = np.concatenate((pid_prbs, prbs), axis=0)

    assert len(pid_prbs) == events_number
    pid_prbs=pid_prbs.astype(float)

    array2tree(pid_prbs, save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--file_path", type=str, help="root file for PID.")
    parser.add_argument("--save_path", type=str, help="root file for saving PID results.")
    parser.add_argument("--model_path", type=str, help="ANN pid model path.")
    parser.add_argument("--n_classes", type=str, default=3, help="set n classes.")
    args = parser.parse_args()

    parameters = {
        'file_path': args.file_path,
        'save_path': args.save_path,
        'model_path': args.model_path,
        'n_classes': args.n_classes,
    }

    PID(**parameters)

    tags=readPIDIndex(args.save_path)

    import matplotlib.pyplot as plt

    plt.hist(tags,log=False)
    plt.title('-1: uncertain, 0:mu+, 1:e+, 2:pi+')
    plt.savefig('/lustre/collider/songsiyuan/CEPC/PID/Calib/pid_result.png')
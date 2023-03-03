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
    simulation = file['T']

    # Get data->numpy
    data = simulation.arrays(library="np")

    # e+,x,y,z
    # hcal_energy = data['Hit_Energy']
    # x = data['Hit_X']
    # y = data['Hit_Y']
    # z = data['Hit_Z']
    hcal_energy = data['hcal_celle']
    x = data['hcal_cellx']
    y = data['hcal_celly']
    z = data['hcal_cellz']

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


def array2tree(array, save_path):
    file = uproot.recreate(save_path)
    file['Calib_Hit'] = {'ANN_PID': array}


def readPIDIndex(file_path):
    file = uproot.open(file_path)
    tree = file['Calib_Hit']

    tree = tree.arrays(library='numpy')
    branch=tree['ANN_PID']
    return branch


def PID(file_path, save_path, model_path, threshold=0, n_classes=3):
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

    tags = None

    with torch.no_grad():
        net.eval()

        for i, inputs in enumerate(pid_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)

            prbs = Softmax(dim=1)(outputs)

            prbs, particles = torch.max(prbs, 1)

            if gpu:
                prbs = prbs.cpu().numpy()
                particles = particles.cpu().numpy()
            else:
                prbs = prbs.numpy()
                particles = particles.numpy()

            batch_size = len(prbs)

            tag_threshold = prbs > threshold

            _ = np.where(tag_threshold, particles, -1 * np.ones(batch_size))

            if i == 0:
                tags = _
            else:
                tags = np.concatenate((tags, _), axis=0)

    assert len(tags) == events_number
    tags=tags.astype(np.int32)

    array2tree(tags, save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--file_path", type=str, help="root file for PID.")
    parser.add_argument("--save_path", type=str, help="root file for saving PID results.")
    parser.add_argument("--model_path", type=str, help="ANN pid model path.")
    parser.add_argument("--threshold", type=float, help="classification threshold, 0-1.")
    parser.add_argument("--n_classes", type=str, default=3, help="set n classes.")
    args = parser.parse_args()

    parameters = {
        'file_path': args.file_path,
        'save_path': args.save_path,
        'model_path': args.model_path,
        'threshold': args.threshold,
        'n_classes': args.n_classes,
    }

    PID(**parameters)

    tags=readPIDIndex(args.save_path)

    import matplotlib.pyplot as plt
    plt.hist(tags,log=False)
    plt.savefig('./test.png')

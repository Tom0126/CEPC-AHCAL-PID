#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 22:12
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : PID.py
# @Software: PyCharm
import pandas as pd
import torch
import numpy as np
from Net.lenet import LeNet_bn as lenet
from Net.resnet import ResNet, BasicBlock, Bottleneck
from torch.nn import Softmax
import uproot
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any
import os


class PIDSet(Dataset):
    def __init__(self, imgs_path, imgs, mean, std
                 , mean_std_static, threshold, transform=None, gap=26) -> None:
        super().__init__()
        # datasets = np.load(img_path)
        # datasets = datasets.astype(np.float32)

        if imgs_path != None:
            img = makeDatasets(imgs_path, tree_name='Calib_Hit', gap=gap, threshold=threshold)
        else:
            img = imgs

        # standardize the train set
        if mean_std_static:
            mean = mean
            std = std
        else:
            mean = np.average(img)
            std = np.std(img)
        img = (img - mean) / std

        self.datasets = img.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index: Any):
        img = self.datasets[index]

        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.datasets)


def pid_data_loader(imgs_path=None,
                    imgs=None,
                    mean=0.0,
                    std=1.0,
                    mean_std_static: bool = True,
                    batch_size: int = 1024,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    gap=26,
                    threshold=0,
                    **kwargs
                    ):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_pid = PIDSet(imgs_path=imgs_path, imgs=imgs, mean=mean, std=std, mean_std_static=mean_std_static,
                         transform=transforms_train, gap=gap, threshold=threshold)

    loader_pid = DataLoader(dataset=dataset_pid, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, drop_last=False)

    return loader_pid


class PIDGNNImageSet(Dataset):
    def __init__(self,
                 imgs,
                 mean,
                 std,
                 mean_std_static,
                 max_nodes,
                 transform=None) -> None:
        super().__init__()

        datasets = imgs

        # standardize the train set
        if mean_std_static:
            mean = mean
            std = std
        else:
            mean = np.average(datasets)
            std = np.std(datasets)
        datasets = (datasets - mean) / std

        self.datasets = datasets.astype(np.float32)

        self.transform = transform

        self.max_nodes = max_nodes

    def __getitem__(self, index: Any):

        img = self.datasets[index]

        img_indices = np.nonzero(img)

        e = img[img_indices[0], img_indices[1], img_indices[2]]
        img = [i.astype(np.float32) for i in img_indices]
        img.append(e)
        img = np.vstack(img)

        if self.max_nodes > img.shape[-1]:

            paddings = np.zeros((img.shape[-2], (self.max_nodes - img.shape[-1])), dtype=np.float32)
            img = np.concatenate([img, paddings], axis=1)

        else:

            choice = np.random.choice(img.shape[-1], self.max_nodes, replace=False)
            img = img[:, choice]



        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.datasets)


def pid_data_loader_gnn(imgs,
                        mean=0.0,
                        std=1.0,
                        mean_std_static: bool = True,
                        batch_size: int = 512,
                        shuffle: bool = False,
                        num_workers: int = 0,
                        max_nodes: int = 1024,
                        **kwargs):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = PIDGNNImageSet(imgs,
                                   mean=mean,
                                   std=std,
                                   mean_std_static=mean_std_static,
                                   transform=transforms_train,
                                   max_nodes=max_nodes
                                   )

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

    return loader_train


class ReadRoot():
    def __init__(self, file_path, tree_name):
        file = uproot.open(file_path)
        tree = file[tree_name]
        self.tree = tree.arrays(library="np")

    def readBranch(self, branch):
        return self.tree[branch]


def readEventNum(file_path, tree_name, save_path=None):
    ahcal = ReadRoot(file_path, tree_name)
    event_num = ahcal.readBranch('Event_Num')
    if save_path != None:
        np.save(save_path, event_num)
        return
    return event_num


def makeDatasets(file_path, tree_name, threshold, save_path=None, gap=26):
    '''
    1: inout: root file
    2: output: numpy array NCHW (,40,18,18)
    Tom's ID: 0:mu+, 1:e+, 2:pi+, 3: noise
    '''

    ahcal = ReadRoot(file_path, tree_name)
    x = ahcal.readBranch('Hit_X')
    y = ahcal.readBranch('Hit_Y')
    z = ahcal.readBranch('Hit_Z')
    e = ahcal.readBranch('Hit_Energy')

    # read raw root file

    num_events = len(e)
    assert num_events == len(x)
    assert num_events == len(y)
    assert num_events == len(z)

    # NHWC
    deposits = np.zeros((num_events, 18, 18, 40))

    for i in range(num_events):

        energies_ = e[i]

        x_ = np.around((x[i] + 342.5491) / 40.29964).astype(int)
        y_ = np.around((y[i] + 343.05494) / 40.29964).astype(int)
        z_ = ((z[i]) / gap).astype(int)
        num_events_ = len(energies_)
        assert num_events_ == len(x_)
        assert num_events_ == len(y_)
        assert num_events_ == len(z_)

        for j in range(num_events_):
            e_ = energies_[j] if energies_[j] > threshold else 0
            deposits[i, x_[j], y_[j], z_[j]] += e_

    # NCHW
    if save_path != None:
        np.save(save_path, deposits)
        return
    return deposits


def array2tree(array, save_path, **kwargs):
    array = np.transpose(array)
    file = uproot.recreate(save_path)

    if 'event_num' in kwargs.keys():
        if len(array) == 4:
            file['Calib_Hit'] = {
                'Event_Num': kwargs.get('event_num'),
                'ANN_mu_plus': array[0],
                'ANN_e_plus': array[1],
                'ANN_pi_plus': array[2],
                'ANN_noise': array[3],
            }
        elif len(array) == 2:
            file['Calib_Hit'] = {
                'Event_Num': kwargs.get('event_num'),
                'ANN_e_plus': array[0],
                'ANN_pi_plus': array[1],

            }
        else:
            file['Calib_Hit'] = {
                'Event_Num': kwargs.get('event_num'),
                'ANN_mu_plus': array[0],
                'ANN_e_plus': array[1],
                'ANN_pi_plus': array[2],

            }
    else:
        if len(array) == 4:
            file['Calib_Hit'] = {

                'ANN_mu_plus': array[0],
                'ANN_e_plus': array[1],
                'ANN_pi_plus': array[2],
                'ANN_noise': array[3],
            }
        elif len(array) == 2:
            file['Calib_Hit'] = {
                'ANN_e_plus': array[0],
                'ANN_pi_plus': array[1],

            }
        else:
            file['Calib_Hit'] = {

                'ANN_mu_plus': array[0],
                'ANN_e_plus': array[1],
                'ANN_pi_plus': array[2],

            }


def readPIDIndex(file_path, threshold=0.9):
    file = uproot.open(file_path)
    tree = file['Calib_Hit']

    tree = tree.arrays(library='numpy')
    mu_prbs = tree['ANN_mu_plus'].reshape(1, -1)
    positron_prbs = tree['ANN_e_plus'].reshape(1, -1)
    pi_prbs = tree['ANN_pi_plus'].reshape(1, -1)
    noise_prbs = tree['ANN_noise'].reshape(1, -1)

    _ = np.concatenate((mu_prbs, positron_prbs), axis=0)
    _ = np.concatenate((_, pi_prbs), axis=0)
    prbs = np.concatenate((_, noise_prbs), axis=0)
    prbs = np.transpose(prbs)  # 0:mu+, 1: e+, 2: pi+. 3: noise:\

    prbs_max = np.amax(prbs, axis=1)

    particles = np.argmax(prbs, axis=1)

    tags = prbs_max > threshold

    num = len(prbs)

    assert num == len(particles)

    pid_results = np.where(tags, particles, -1 * np.ones(num))

    return pid_results


def PID(file_path, save_path, model_path, n_classes, net_used, net_dict, net_para_dict, hit_threshold, z_gap=26):
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

    net = net_dict.get(net_used)
    net_paras = net_para_dict.get(net_used)
    net = net(**net_paras)

    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    event_number = readEventNum(file_path, tree_name='Calib_Hit')

    pid_loader = pid_data_loader(imgs_path=file_path, batch_size=2048, gap=z_gap, threshold=hit_threshold)

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

    pid_prbs = pid_prbs.astype(float)

    array2tree(array=pid_prbs, save_path=save_path, event_num=event_number)


def npyPID(file_path,
           save_path,
           model_path,
           net,
           pid_data_loader_func,
           max_nodes,
           df=False,
           **kwargs
           ):
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

    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    pid_loader = pid_data_loader_func(imgs=np.load(file_path, allow_pickle=True),
                                      batch_size=128,
                                      max_nodes=max_nodes)

    pid_prbs = []

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

            pid_prbs.append(prbs)

        pid_prbs = np.concatenate(pid_prbs, axis=0)

    # assert len(pid_prbs) == events_number
    pid_prbs = pid_prbs.astype(float)

    if df:
        df = {}
        for i, col in enumerate(kwargs.get('cols')):
            df[col] = pid_prbs[:, i]

        if 'labels' in kwargs.keys():
            labels_ = kwargs.get('labels')
            df['particle_label'] = labels_

        df = pd.DataFrame(df)
        df.to_csv(save_path, index=False)
    else:
        array2tree(array=pid_prbs, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--file_path", type=str, help="root file for PID.")
    parser.add_argument("--npy_path", type=str, help="npyfile for PID.")
    parser.add_argument("--tree", type=str, default='Calib_Hit', help="tree name.")
    parser.add_argument("--save_path", type=str, help="root file for saving PID results.")
    parser.add_argument("--model_path", type=str, help="ANN pid model path.")
    parser.add_argument("--n_classes", type=str, default=4, help="set n classes.")
    parser.add_argument("--z_gap", type=int, default=26)
    parser.add_argument("--hit_threshold", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        'file_path': args.file_path,
        'save_path': args.save_path,
        'model_path': args.model_path,
        'n_classes': args.n_classes,
        'z_gap': args.z_gap,
        'hit_threshold': args.hit_threshold,

    }
    PID(**parameters)
    # makeDatasets(file_path=args.file_path,tree_name=args.tree,save_path=args.npy_path)
    # npyPID(file_path=args.npy_path,save_path=args.save_path,model_path=args.model_path,n_classes=args.n_classes)

    print('DONE')
    #
    # tags=readPIDIndex(args.save_path)
    # makeDatasets(args.file_path,save_path='/lustre/collider/songsiyuan/CEPC/PID/TB2NPY/AHCAL_Run131_e.npy')
    # import matplotlib.pyplot as plt
    #
    # plt.hist(tags,log=False)
    # plt.title('-1: uncertain, 0:mu+, 1:e+, 2:pi+')
    # plt.savefig('/lustre/collider/songsiyuan/CEPC/PID/Calib/pid_result.png')

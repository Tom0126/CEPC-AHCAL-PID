#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/19 14:40
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : dgcnn.py
# @Software: PyCharm

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Data.loader import data_loader_gnn

# a copy and adjustment based on "Dynamic Graph CNN for Learning on Point Clouds" from
# https://github.com/WangYueFt/dgcnn, https://github.com/antao97/dgcnn.pytorch/tree/master

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)

    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class PCNN(nn.Module):
    def __init__(self,
                 k: int,
                 in_channel: int,
                 channels: list,
                 kernels: list,
                 bns: list,
                 acti: list,
                 get_f_func,
                 adaptive_pool: bool,
                 pool_out_size: tuple = None
                 ):

        super(PCNN, self).__init__()

        self.k = k
        self.adaptive_pool = adaptive_pool
        self.pool_out_size = pool_out_size
        self.get_f_func = get_f_func

        self.num_layers = len(channels)

        assert self.num_layers == len(kernels)
        assert self.num_layers == len(bns)

        self.convs = nn.ModuleList()

        for i in range(self.num_layers):

            layers = []

            in_c = 2 * in_channel if i == 0 else channels[i - 1] * 2

            out_c = channels[i]

            layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernels[i], bias=False))

            if bns[i]:
                layers.append(nn.BatchNorm2d(out_c))

            if acti[i]:
                layers.append(nn.LeakyReLU(negative_slope=0.2))

            self.convs.append(nn.Sequential(*layers))



    def forward(self, x):

        x = torch.squeeze(x, dim=1)

        for i, conv in enumerate(self.convs):

            x = self.get_f_func(x, k=self.k)

            x = conv(x)  # (-1, C, P, k))

            if i < (self.num_layers - 1):
                x = x.max(dim=-1, keepdim=False)[0]

        if self.adaptive_pool:
            x = nn.AdaptiveMaxPool2d(self.pool_out_size)(x)

        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()
class DGCNN_cls(PCNN):

    def __init__(self,
                 k: int,
                 in_channel: int,
                 channels: list,
                 emb_dims: int,
                 num_classes: int,
                 kernels: list,
                 bns: list,
                 acti: list,
                 get_f_func,
                 adaptive_pool: bool,
                 dropout: float,
                 pool_out_size: tuple = None):
        super().__init__(k, in_channel, channels, kernels, bns, acti, get_f_func, adaptive_pool, pool_out_size)

        self.conv1d = nn.Sequential(nn.Conv1d(sum(channels), emb_dims, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(emb_dims),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x_list = []

        for i, conv in enumerate(self.convs):
            x = self.get_f_func(x, k=self.k)
            x = conv(x)  # (-1, C, P, k))
            x = x.max(dim=-1, keepdim=False)[0]
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.conv1d(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(x.shape[0],
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(x.shape[0],
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x


if __name__ == '__main__':


    pass

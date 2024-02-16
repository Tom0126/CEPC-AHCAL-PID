#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 14:38
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : dgres.py
# @Software: PyCharm

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/3 10:24
# @Author  : Tom SONG
# @Mail    : xdmyssy@gmail.com
# @File    : resnet.py
# @Software: PyCharm


from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from Data.loader import data_loader_gnn
from Net.dgcnn import PCNN, get_graph_feature
from Net.resnet import ResNet, ResNet_Avg, Bottleneck, BasicBlock


class DGRes(nn.Module):

    def __init__(self,
                 PCNN_block: nn.Module,
                 k: int,
                 in_channel: int,
                 channels: list,
                 kernels: list,
                 bns: list,
                 acti: list,
                 get_f_func,
                 adaptive_pool: bool,
                 pool_out_size: tuple,
                 Res_block: nn.Module,
                 block: nn.Module,
                 layers: list,
                 start_planes=40,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 short_cut=True,
                 first_kernal=7,
                 first_stride=2,
                 first_padding=3,

                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pcnn = PCNN_block(k=k,
                               in_channel=in_channel,
                               channels=channels,
                               kernels=kernels,
                               bns=bns,
                               acti=acti,
                               get_f_func=get_f_func,
                               adaptive_pool=adaptive_pool,
                               pool_out_size=pool_out_size
                               )
        self.resnet = Res_block(block=block,
                                layers=layers,
                                num_classes=num_classes,
                                start_planes=start_planes,
                                short_cut=short_cut,
                                first_kernal=first_kernal,
                                first_stride=first_stride,
                                first_padding=first_padding,
                                zero_init_residual=zero_init_residual,
                                groups=groups,
                                width_per_group=width_per_group,
                                replace_stride_with_dilation=replace_stride_with_dilation,
                                norm_layer=norm_layer,

                                )

    def forward(self, x):
        x = self.pcnn(x)
        x = self.resnet(x)

        return x


if __name__ == '__main__':



    pass


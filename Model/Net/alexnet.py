#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/7 02:33
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : alexnet.py
# @Software: PyCharm

# adjustment on https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

import torch
import torch.nn as nn
from Data.loader import data_loader
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self, n_classes: int = 1000, dropout: float = 0.5, in_channel:int=40) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':


    pass

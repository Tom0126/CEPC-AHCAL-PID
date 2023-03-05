import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


img_shape=(11,21,21)

class Generator(nn.Module):
    def __init__(self,latent_dim,n_classes):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_fc = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 2*2*128, normalize=True),
            # *block(256, 512),
            # *block(256, 1600),
        )
        self.model_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4*4*128
            nn.Conv2d(128, 64, 3, padding=(1, 1)),  # 4*4*64
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 8*8*64
            nn.Conv2d(64, 32, 3, padding=(1, 1)),  # 8*8*32
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 16*16*32
            nn.Conv2d(32, 16, 3, padding=(1, 1)),  # 16*16*16
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 16, 3, padding=(2, 2)),  # 18*18*16
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 16, 3, padding=(2, 2)),  # 20*20*16
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 16, 3, padding=(2, 2)),  # 22*22*16
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 11, 3, padding=(1, 1)),  # 21*21*11

            nn.LeakyReLU(0, inplace=True),

        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model_fc(gen_input)
        img = img.view(img.size(0), 64, 5, 5)

        img = self.model_conv(img)

        return img

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self,n_classes):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes)

        # self.model_conv=nn.Sequential(
        #
        #     nn.Conv2d(11, 32, 3, padding=(0, 0)),  # 19*19*32
        #     nn.BatchNorm2d(num_features=32),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(32, 64, 3, padding=(0, 0)),  # 17*17*64
        #     nn.BatchNorm2d(num_features=64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.MaxPool2d(2,2), # 9*9*64
        #
        #     nn.Conv2d(64, 128, 3, padding=(0, 0)),  # 7*7*128
        #     nn.BatchNorm2d(num_features=128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.MaxPool2d(2, 2),  # 4*4*128
        #
        #     nn.Conv2d(128, 256, 3, padding=(1, 1)),  # 4*4*256
        #     nn.BatchNorm2d(num_features=256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        # )
        self.model_fc = nn.Sequential(
            nn.Linear(self.n_classes + int(np.prod(img_shape)), 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1024),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model_fc(d_in)
        return validity

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(64,10).to(device)
    gen.initialize_weights()
    z = torch.Tensor(np.random.normal(0, 1, (2, 64)))
    label = torch.Tensor([0, 9]).long()
    imgs = gen(z, label)
    print(z.shape,label.shape,imgs.shape)
    te=np.sum(imgs.detach().cpu().numpy(),axis=(1,2,3))
    dis = Discriminator(10).to(device)
    dis.initialize_weights()
    validity = dis(imgs, label)
    print(validity)
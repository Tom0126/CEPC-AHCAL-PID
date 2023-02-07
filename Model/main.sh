#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /sw/anaconda/3.8-2021.05/thisconda.sh
conda activate pytorch

# change hyper-parameters

n_epoch=300
batch_size=32
lr=0.001
mean=0.07
std=1.63
optim='SGD'
n_classes=4

python /home/songsiyuan/CEPC/PID/Model/Train.py --n_epoch $n_epoch -b $batch_size -lr $lr --mean $mean --std $std --optim $optim --n_classes $n_classes


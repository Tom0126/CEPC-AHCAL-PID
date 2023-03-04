#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /sw/anaconda/3.8-2021.05/thisconda.sh
conda activate pytorch

file_path=/lustre/collider/songsiyuan/CEPC/PID/Calib/mu+/AHCAL_Run102_20221023_114706.root
save_path=/lustre/collider/songsiyuan/CEPC/PID/Calib/AHCAL_Run102_ANN_PID.root
model_path=/lustre/collider/songsiyuan/CEPC/PID/CheckPoint/epoch_50_lr_0.001_batch_32_mean_0.07_std_1.62_optim_Adam/net.pth



python /home/songsiyuan/CEPC/PID/Model/PID.py --file_path $file_path --save_path $save_path --model_path $model_path


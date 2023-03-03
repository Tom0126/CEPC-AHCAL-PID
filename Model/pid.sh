#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /sw/anaconda/3.8-2021.05/thisconda.sh
conda activate pytorch

file_path=/lustre/collider/songsiyuan/CEPC/PID/Calib/AHCAL_Run50_20221021_050946.root
save_path=/lustre/collider/songsiyuan/CEPC/PID/Calib/AHCAL_Run50_ANN_PID.root
model_path=/lustre/collider/songsiyuan/CEPC/PID/CheckPoint/epoch_300/net.pth



python /home/songsiyuan/CEPC/PID/Model/PID.py --file_path $file_path --save_path $save_path --model_path $model_path


#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /sw/anaconda/3.8-2021.05/thisconda.sh
conda activate pytorch

file_path=/lustre/collider/xuzixun/software/siyuancalo/cepc-calo/run/pi+_10k/ahcal_pi+_70GeV_2cm2cm_10k.root
save_path=./test.root
model_path=/lustre/collider/songsiyuan/CEPC/PID/CheckPoint/epoch_300/net.pth
threshold=0.9


python /home/songsiyuan/CEPC/PID/Model/PID.py --file_path $file_path --save_path $save_path --model_path $model_path --threshold $threshold


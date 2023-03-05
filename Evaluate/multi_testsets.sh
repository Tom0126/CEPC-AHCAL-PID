#!/bin/bash
#source /sw/anaconda/3.7-2020.02/thisconda.sh
#conda activate testenv

source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh

energy_point=100


python /home/songsiyuan/CEPC/PID/Evaluate/MakeFinalDataset.py --ep $energy_point
#python MakeTrainsets.py
#python Test.py

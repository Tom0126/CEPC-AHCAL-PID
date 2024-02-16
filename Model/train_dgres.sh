#! /bin/bash

source #<AN ENV You>
conda activate #<ENV>

# change hyper-parameters
n_classes=2
n_epoch=200
batch_size=1024
lr=0.1
optim='SGD'
max_nodes=64
k=20git
l_gamma=1
step=100
short_cut=1
f_k=3
f_s=1
f_p=1
index=0
debug=0
lr_schedule='step'


python ./main_dgres.py \
--n_epoch $n_epoch \
-b $batch_size \
-lr $lr \
--optim $optim \
--n_classes $n_classes \
--index $index \
--max_nodes $max_nodes \
--k $k \
--l_gamma $l_gamma \
--step $step \
--short_cut $short_cut \
--f_k $f_k \
--f_s $f_s \
--f_p $f_p \
--debug $debug \
--lr_schedule $lr_schedule


#! /bin/bash


source #<AN ENV>
conda activate #<ENV NAME>

# change hyper-parameters

n_epoch=200
batch_size=128
lr=0.0001
optim='SGD'
n_classes=2
l_gamma=1
step=100
index=0
debug=1
val_interval=1
eval=0
ana_eval=0
lr_schedule='step'




python ./main_alexnet.py \
--n_epoch $n_epoch \
-b $batch_size \
-lr $lr \
--optim $optim \
--n_classes $n_classes \
--l_gamma $l_gamma \
--step $step \
--index $index \
--debug $debug \
--val_interval $val_interval \
--eval $eval \
--ana_eval $ana_eval \
--lr_schedule $lr_schedule



#! /bin/bash


source #<AN ENV>
conda activate #<ENV NAME>

# change hyper-parameters

n_epoch=200
batch_size=32
lr=0.0001
optim='SGD'
n_classes=2
l_gamma=1
step=100
short_cut=0
f_k=3
f_s=1
f_p=1
index=0
debug=0
val_interval=1
eval=0
ana_eval=0
lr_schedule='step'
train=1




python ./main_resnet.py \
--n_epoch $n_epoch \
-b $batch_size \
-lr $lr \
--optim $optim \
--n_classes $n_classes \
--l_gamma $l_gamma \
--step $step \
--short_cut $short_cut \
--f_k $f_k \
--f_s $f_s \
--f_p $f_p \
--index $index \
--debug $debug \
--val_interval $val_interval \
--eval $eval \
--ana_eval $ana_eval \
--lr_schedule $lr_schedule \
--train $train



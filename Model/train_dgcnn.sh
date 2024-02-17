#! /bin/bash

source #<AN ENV>
conda activate #<ENV NAME>

# change hyper-parameters
n_classes=2
n_epoch=200
batch_size=1024
lr=0.1
optim='SGD'
max_nodes=64
k=20
index=0
debug=0



python ./main_dgcnn.py \
--n_epoch $n_epoch \
-b $batch_size \
-lr $lr \
--optim $optim \
--n_classes $n_classes \
--index $index \
--max_nodes $max_nodes \
--k $k \
--debug $debug


#! /bin/bash

lr=0.000001
optim=('SGD' 'Adam')
batch=(512 128 32)
epoch=(50)




num=0

for((i=0;i<3;i++))
do

lr=$(echo "$lr * 10" | bc -l)

for((j=0;j<${#epoch[@]};j++))
do

for((k=0;k<${#batch[@]};k++))
do

for((m=0;m<${#optim[@]};m++))
do



  cp main.sh scripts/main_$num.sh
  cp a100_train.sub scripts/a100_train_$num.sub

  sed -i 2cExecutable\=./scripts/main_$num.sh scripts/a100_train_$num.sub

  sed -i 10cn_epoch\=${epoch[j]} scripts/main_$num.sh
  sed -i 11cbatch_size\=${batch[k]} scripts/main_$num.sh
  sed -i 12clr\=$lr scripts/main_$num.sh
  sed -i 13coptim\=${optim[m]} scripts/main_$num.sh

  condor_submit scripts/a100_train_$num.sub

  num=`expr $num + 1`

done
done
done
done



#!/bin/bash

#NET="ResNet-50"
NET="ResNet-101"
iter=30000
model7="resnet101_finetune_7_iter_${iter}.caffemodel"
model30="resnet101_finetune_30_iter_${iter}.caffemodel"
model13="resnet101_finetune_13_iter_${iter}.caffemodel"
if [[ $1 == "base" ]]
then
  caffemodel=$model7
elif [[ $1 == "class" ]]
then
  caffemodel=$model30
elif [[ $1 == "color" ]]
then
  caffemodel=$model13
else
  echo "usage: ./rfcn_run.sh [base|class|color] [gpu id]"
  exit 1
fi
name=$1
if [[ $2 == "" ]]
then
  echo "usage: ./rfcn_run.sh [base|class|color] [gpu id]"
  exit 1
fi
GPU_ID=$2

./tools/detect_rfcn.py \
  --gpu ${GPU_ID} \
  --model $caffemodel \
  --name ${name} \
  --net $NET

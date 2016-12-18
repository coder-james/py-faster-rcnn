#!/bin/bash
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2016 Peking University
# Licensed under The MIT License [see LICENSE for details]
# Written by coder-james
# --------------------------------------------------------

pt_dir="pascal_voc"
net1="VGG16"
ITERS="8000"
solver1="solver7.prototxt"
solver2="solver30.prototxt"
solver3="solver13.prototxt"
if [[ $1 == "" || $2 == "" ]] 
then
  echo "usage:run.sh [train/test] [base/class/color]"
  exit 1
fi
if [[ $2 == "base" ]]
then 
  solver=${solver1}
  TRAIN_IMDB=$2
elif [[ $2 == "class" ]]
then 
  solver=${solver2}
  TRAIN_IMDB=$2
elif [[ $2 == "color" ]]
then 
  solver=${solver3}
  TRAIN_IMDB=$2
else 
  echo "illegal arg!"
  exit 1
fi
GPUid=$3
testproto7="test7.prototxt"
testproto30="test30.prototxt"
testproto13="test13.prototxt"
imagelist="output/test_imagelist.txt"
imagedir="test/images"
savefile="output/detect_"$2
if [[ $1 == "train" ]]
then
./tools/train_net.py \
  --gpu ${GPUid} \
  --weights data/${net1}_faster_rcnn_final.caffemodel \
  --solver models/${pt_dir}/${net1}/faster_rcnn_end2end/${solver} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml 
else
./tools/detect.py \
  --gpu ${GPUid} \
  --prototxt models/${pt_dir}/${net1}/faster_rcnn_end2end/${testproto} \
  --caffemodel output/${caffemodel} \
  --imagelist data/ccf/${imagelist} \
  --imagedir data/ccf/${imagedir} \
  --type $2 \
  --savefile data/ccf/${savefile}
fi

#!/bin/bash

#NET="ResNet-50"
NET="ResNet-101"
ITERS=60000

SOLVER7="solver_ohem7.prototxt"
SOLVER30="solver_ohem30.prototxt"
SOLVER13="solver_ohem13.prototxt"
model7="resnet101_rfcn_ohem_7_iter_120000.caffemodel"
model30="resnet101_rfcn_ohem_30_iter_120000.caffemodel"
model13="resnet101_rfcn_ohem_13_iter_120000.caffemodel"
if [[ $1 == "base" ]]
then
  solver=$SOLVER7
  caffemodel=$model7
elif [[ $1 == "class" ]]
then
  solver=$SOLVER30
  caffemodel=$model30
elif [[ $1 == "color" ]]
then
  solver=$SOLVER13
  caffemodel=$model13
else
  echo "usage: ./rfcn_run.sh [base|class|color] [gpu id]"
  exit 1
fi
TRAIN_IMDB=$1
if [[ $2 == "" ]]
then
  echo "usage: ./rfcn_run.sh [base|class|color] [gpu id]"
  exit 1
fi
GPU_ID=$2

LOG="experiments/logs/rfcn_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --gpu ${GPU_ID} \
  --solver models/${NET}/rfcn_end2end/${solver} \
  --weights data/imagenet_models/${caffemodel} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/rfcn_end2end_ohem.yml
#  --weights data/imagenet_models/resnet50_rfcn_ohem7_iter_110000.caffemodel \

#time ./tools/test_net.py --gpu ${GPU_ID} \
#  --def models/${PT_DIR}/${NET}/rfcn_end2end/test_agnostic.prototxt \
#  --net ${NET_FINAL} \
#  --imdb ${TEST_IMDB} \
#  --cfg experiments/cfgs/rfcn_end2end_ohem.yml \
#  ${EXTRA_ARGS}

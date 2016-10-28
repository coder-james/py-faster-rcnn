# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
  ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
# R-FCN: Object Detection via Region-based Fully Convolutional Networks

py-R-FCN now supports joint training. 

py-R-FCN is based on the [py-faster-rcnn code](https://github.com/rbgirshick/py-faster-rcnn )(include this README) and [the offcial R-FCN implementation](https://github.com/daijifeng001/R-FCN), and the usage is quite similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn ), thanks for their great works.

#### Some modification

The original py-faster-rcnn uses class-aware bounding box regression. However, R-FCN use class-agonistic bounding box regression to reduce model complexity. So I add a configuration AGONISTIC into fast_rcnn/config.py, and the default value is False. You should set it to True both on train and test phase if you want to use class agonistic training and test. 

OHEM need all rois to select the hard examples, so I changed the sample strategy, set `BATCH_SIZE: -1` for OHEM, otherwise OHEM would not take effect.

In conclusion:

`AGONISTIC: True` is required for class-agonistic bounding box regression

`BATCH_SIZE: -1` is required for OHEM

And I've already provided two configuration files for you(w/ OHEM and w/o OHEM) under `experiments/cfgs` folder, you could just use them and needn't change anything.
    
### Main Results
                   | training data       | test data             | mAP   | time/img (K40) | time/img (Titian X)
-------------------|:-------------------:|:---------------------:|:-----:|:--------------:|:------------------:|
R-FCN, ResNet-50  | VOC 07+12 trainval  | VOC 07 test           | 76.9%(80k110k) | -        | 0.099sec            |
R-FCN, ResNet-101 | VOC 07+12 trainval  | VOC 07 test           | 78.7%(80k110k) | -        | 0.136sec           |


### Requirements: software

0. **`Important`** Please use the [Microsoft-version Caffe(@commit 1a2be8e)](https://github.com/Microsoft/caffe/tree/1a2be8ecf9ba318d516d79187845e90ac6e73197), this Caffe supports R-FCN layer, and the prototxt in this repository follows the Microsoft-version Caffe's layer name. You need to put the Caffe root folder under py-R-FCN folder, just like what py-faster-rcnn does.

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

Any NVIDIA GPU with 6GB or larger memory is OK(4GB is enough for ResNet-50).


### Installation
1. Clone the R-FCN repository
  ```Shell
  git clone https://github.com/Orpine/py-R-FCN.git
  ```
  We'll call the directory that you cloned R-FCN into `RFCN_ROOT`

2. Clone the Caffe repository
  ```Shell
  cd $RFCN_ROOT
  git clone https://github.com/Microsoft/caffe.git
  ```
  [optional] 
  ```Shell
  cd caffe
  git reset --hard 1a2be8e
  ```
  (I only test on this commit, and I'm not sure whether this Caffe is still compatible with the prototxt in this repository in the future)
  
  If you followed the above instruction, python code will add `$RFCN_ROOT/caffe/python` to `PYTHONPATH` automatically, otherwise you need to add `$CAFFE_ROOT/python` by your own, you could check `$RFCN_ROOT/tools/_init_paths.py` for more details.

3. Build the Cython modules
    ```Shell
    cd $RFCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $RFCN_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
   ```

### Demo
1.  To use demo you need to download the pretrained R-FCN model, please download the model manually from [OneDrive](https://1drv.ms/u/s!AoN7vygOjLIQqUWHpY67oaC7mopf), and put it under `$RFCN/data`. 

    Make sure it looks like this:
    ```Shell
    $RFCN/data/rfcn_models/resnet50_rfcn_final.caffemodel
    $RFCN/data/rfcn_models/resnet101_rfcn_final.caffemodel
    ```

2.  To run the demo
  
    ```Shell
    $RFCN/tools/demo_rfcn.py
    ```
    
  The demo performs detection using a ResNet-101 network trained for detection on PASCAL VOC 2007.

### Usage

To train and test a R-FCN detector using the **approximate joint training** method, use `experiments/scripts/rfcn_end2end.sh`.
Output is written underneath `$RFCN_ROOT/output`.

To train and test a R-FCN detector using the **approximate joint training** method **with OHEM**, use `experiments/scripts/rfcn_end2end_ohem.sh`.
Output is written underneath `$RFCN_ROOT/output`.

```Shell
cd $RFCN_ROOT
./experiments/scripts/rfcn_end2end[_ohem].sh [GPU_ID] [NET] [DATASET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ResNet-50, ResNet-101} is the network arch to use
# DATASET in {pascal_voc, coco} is the dataset to use(I only tested on pascal_voc)
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Trained R-FCN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

### Misc
py-faster-rcnn code can also work properly, but I do not add any other feature(such as ResNet and OHEM).

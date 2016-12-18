#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Updated by coder-james
# --------------------------------------------------------

"""
Script showing detections in images.

"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import datasets.ccf as ccf
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


def detect(net, name, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'ccf','test','images',image_name)
    im = cv2.imread(im_file)
    shape = "%s,%s,%s" % (im.shape[1], im.shape[0], im.shape[2])

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0
    NMS_THRESH = 0.3
    CLASSES = ccf.getClasses(name)
    content = ""
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:,-1] >= CONF_THRESH)[0]
        tmp = ""
        #inds = 1 if cls != "shoes" else 2
        for i in inds:
          temp = dets[i]
          tmp += "%s|" % ",".join(map(str,temp))
        #for det in dets:
        #  tmp += "%s|" % ",".join(map(str,det))
        if tmp != "":
          content += "%s:%s;" %(cls, tmp[:-1])
    return "%s$%s$%s" % (image_name, shape,content[:-1])
#        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--name', dest='name', help='class name',
                        default='base')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        default='ResNet-50')
    parser.add_argument('--model', dest='model', help="caffemodel to test")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
    name = args.name
  
    prototxt = os.path.join('models', args.demo_net,
                            'rfcn_end2end', ccf.getTestPF(name))
    caffemodel = os.path.join('output', 'rfcn_end2end_ohem', 'ccf',
                              args.model)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    with open(os.path.join(cfg.DATA_DIR, 'ccf', 'output', "test_imagelist.txt")) as imfile:
      im_names = ["IMG_" + item + ".jpg" for item in imfile.read().split("\n") if len(item) > 1]
    timer = Timer()
    timer.tic()
    content=""
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    for im_name in im_names:
        content += detect(net, name, im_name) + "\n"
    timer.toc()
    print ('Detection took {:.3f}s').format(timer.total_time)
    with open("output_" + name + "_" + args.model.split(".")[0].split("_")[-1] + ".txt", "w") as detectfile:
      detectfile.write(content[:-1])

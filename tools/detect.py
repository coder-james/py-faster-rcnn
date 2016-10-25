#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2016 Peking University
# Licensed under The MIT License [see LICENSE for details]
# Written by coder-james
# --------------------------------------------------------

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

#CLASSES = ('__background__','woman', 'longhair', 'downjacket', 'gray', 'pants', 'black', 'singleshoulder', 'brown', 'coat', 'red', 'green', 'boots', 'man', 'shorthair', 'multicolor', 'otherbag', 'yellow', 'hat', 'backpack', 'blue', 'skirt', 'leathershoes', 'otherhair', 'purple', 'white', 'orange', 'sneakers', 'othercolor', 'othertop', 'maxiskit', 'bag', 't-shirt', 'blouse', 'western', 'shorts', 'otherdown', 'othershoes', 'handbox', 'sandal', 'otherman', 'wallet')
#CLASSES = ('__background__','woman', 'longhair', 'downjacket', 'pants', 'singleshoulder', 'coat', 'boots', 'man', 'shorthair','otherbag', 'hat', 'backpack', 'skirt', 'leathershoes', 'otherhair', 'sneakers', 'othertop', 'maxiskit', 'bag', 't-shirt', 'blouse', 'western', 'shorts', 'otherdown', 'othershoes', 'handbox', 'sandal', 'otherman', 'wallet')
#CLASSES = ('__background__','black', 'white', 'red', 'yellow', 'blue', 'green', 'purple', 'brown', 'gray', 'orange', 'multicolor', 'othercolor')
CLASSES = ('__background__','head','top','down','bag','hat','shoes')
#shoes_classes=('leathershoes','sneakers','sandal','boots','othershoes')
#color_classes=('black','white','red','yellow','blue','green','purple','brown','gray','orange','multicolor','othercolor')

def demo(net, image_dir, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    im_file = os.path.join(image_dir, image_name)
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
    content = ""
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #if cls in shoes_classes:
        #  inds = 2
        #elif cls in color_classes:
        #  inds = 5
        #else:
        #  inds = 1
        #inds = np.where(dets[:,-1] >= CONF_THRESH)[0]
        tmp = ""
        inds = 1 if cls != "shoes" else 2
        for i in range(inds):
          temp = dets[i]
          tmp += "%s|" % ",".join(map(str,temp))
        #for det in dets:
        #  tmp += "%s|" % ",".join(map(str,det))
        if tmp != "":
          content += "%s:%s;" %(cls, tmp[:-1])
    return "%s$%s$%s" % (image_name, shape,content[:-1])

def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Detection')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--prototxt', dest='prototxt', help='test prototxt',
                        default=None, type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffe model to use',
                        default=None, type=str)
    parser.add_argument('--imagelist', dest='imagelist', help='image list file',
                        default=None, type=str)
    parser.add_argument('--imagedir', dest='imagedir', help='image data directory',
                        default=None, type=str)
    parser.add_argument('--savefile', dest='savefile', help='save the result',
                        default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    with open(args.imagelist) as imagelist:
      im_names = imagelist.read().split("\n")

    print "load %s test images " % len(im_names)
    timer = Timer()
    timer.tic()
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    content = ""
    for im_name in im_names:
        print 'Test for {:s}'.format("IMG_" + im_name + ".jpg")
        content += demo(net, args.imagedir, "IMG_" + im_name + ".jpg") + "\n"
    timer.toc()
    print ('Detection took {:.3f}s').format(timer.total_time)
    with open(args.savefile, "w") as detectfile:
      detectfile.write(content[:-1])

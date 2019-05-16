from __future__ import print_function

import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import test_model, test_image, export_onnx_model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    parser.add_argument('--onnx', dest='onnx_file',
                    help='optional onnx_file to be exported', default=None, type=str)
    parser.add_argument('--single_image', dest='single_image',
                        help='inference objects from image', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def test():
    test_model()

def test_single_image(image_path):
    image = cv2.imread(image_path)
    test_image(image)
    cv2.imwrite('/tmp/np_result.jpg', image)

if __name__ == '__main__':
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)

    if args.onnx_file is not None:
        export_onnx_model(args.onnx_file)

    elif args.single_image is not None:
        test_single_image(args.single_image)
        #export_onnx_model("/tmp/bayer_ssd_lite_mbv2.onnx")
        #test_single_image('/mnt/500GB/datasets/Bayer_0315/photos/1.jpg')
        #test_single_image('/tmp/900x600.jpg')
    else:
        test_model()

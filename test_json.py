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
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import test_model, test_image, export_onnx_model,create_npjson,init_checkpoint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    parser.add_argument('--onnx', dest='onnx_file',
                    help='optional onnx_file to be exported', default=None, type=str)
    parser.add_argument('--output', dest='out_put',
                    help='out put path', default=None, type=str)
    parser.add_argument('--testpath', dest='test_path',
                    help='out test dataset path', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def test():
    test_annos_temp = dict()
    s = init_checkpoint()
    examples_list = [f for f in os.listdir(args.test_path) if os.path.isfile(os.path.join(args.test_path, f))]
    json_path = args.test_path + "../templates.json"
    for idx, example in enumerate(examples_list):
        image = cv2.imread(os.path.join(args.test_path, example))
        im_height, im_width, _ = image.shape
        print(example)
        bbjson = create_npjson(s,image,json_path,True)
        test_annos_temp = {
            "version": "3.0.0",
            "company": "RB_Part2",
            "dataset": "Photos",
            "filename": example,
            "image_width": im_width,
            "image_height": im_height,
            "bndboxes": bbjson}
        #print(test_annos_temp)
        fd = open(args.out_put + example + '.json', 'w')
        json.dump(test_annos_temp, fd)
        fd.close()



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
    else:
        test()
        #export_onnx_model("/tmp/bayer_ssd_lite_mbv2.onnx")
        #test_model()
        #test_single_image('/mnt/500GB/datasets/Bayer_0315/photos/1.jpg')

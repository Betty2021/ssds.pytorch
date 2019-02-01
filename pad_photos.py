import argparse
import cv2
import json
from shutil import copytree
import os
import sys
import fnmatch
from os.path import abspath, join, splitext, basename
import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import random

def pad_images():
    parser = argparse.ArgumentParser()

    parser.add_argument("--photo_folder", default="/Users/keyong/Documents/ssd/datasets/PMIUkrainePreUAT/photos", type=str, help="where the image files are ")
    args = parser.parse_args()
    photos_path=args.photo_folder
    all_files = os.listdir(args.photo_folder)

    photo_files = []
    for file in all_files:
        if not file.startswith("."):
            if fnmatch.fnmatch(file, '*.jpg') \
                    or fnmatch.fnmatch(file, '*.JPG'):
                photo_files.append(os.path.join(photos_path, file))

    for img_path in photo_files:
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        h0,w0,_=img.shape
        h1 = int(w0*800.0/600)
        w1 = int(h0*600.0/800)

        if h1>h0+10: # too fat
            print("right padding image:"+img_path)
            img = cv2.copyMakeBorder(img, 0, h1-h0, 0, 0,cv2.BORDER_CONSTANT, 0)
            cv2.imwrite(img_path,img)
        elif w1>w0+10: # too narrow
            print("bottom padding image:"+img_path)
            img = cv2.copyMakeBorder(img, 0, 0, 0, w1-w0 ,cv2.BORDER_CONSTANT, 0)
            cv2.imwrite(img_path,img)





if __name__ == '__main__':
    pad_images()

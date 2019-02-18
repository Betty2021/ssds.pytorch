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

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder",
                    type=str,
                    #default="/Users/keyong/Documents/anno/public/working/PMIUkraine_UAT_validation",
                    #default="/Users/keyong/Documents/anno/public/working/PMIUkrainePreUAT",
                    #default="/home/keyong/Downloads/anno/public/working/field_test_20181004",
                    #default="/home/keyong/Downloads/anno/public/working/harpic_20181113",
                    default="/Users/keyong/Documents/anno/public/working/harpic_20181113",
                    #default="/Users/keyong/Documents/anno/public/working/RB_Total_train_Flag",
                    help="put entire directory")
parser.add_argument("--sku_id",
                    type=str,
                    #default="/Users/keyong/Documents/anno/public/working/PMIUkraine_UAT_validation",
                    #default="/Users/keyong/Documents/anno/public/working/PMIUkrainePreUAT",
                    #default="/home/keyong/Downloads/anno/public/working/field_test_20181004",
                    default="H_competitor",
                    #default="/Users/keyong/Documents/anno/public/working/RB_AD_test",
                    #default="/Users/keyong/Documents/anno/public/working/RB_Total_train_Flag",
                    help="the sku id of box you want to remove")

aargs = parser.parse_args()

DEBUG = False


def get_photos_path(out_folder):
    photos_path = join(out_folder, 'photos')
    if not os.path.exists(photos_path):
        photos_path = join(out_folder, 'Photos')
    assert os.path.exists(photos_path), "photo folder is not found at {}".format(photos_path)
    return photos_path


def find_annotation_files(photos_path):
    annotation_path = join(photos_path, "Annotations")
    assert os.path.exists(annotation_path), "Annotation folder is not found at {}".format(annotation_path)
    all_files = os.listdir(annotation_path)
    annotation_files = []
    for file in all_files:
        if not file.startswith("."):
            if fnmatch.fnmatch(file, '*.json') or fnmatch.fnmatch(file, '*.JSON'):
                annotation_files.append(os.path.join(annotation_path, file))
    annotation_files = sorted(annotation_files)
    return annotation_files


def find_files(out_folder):
    photos_path = get_photos_path(out_folder)
    annotation_files = find_annotation_files(photos_path)
    return annotation_files


target_w = 900
target_h = 1200



#def clean_contradictions(json_data, json_path):
def remove_sku(json_path):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    dirty = False
    bndboxes = json_data.get("bndboxes")
    bndboxes[:] = [ box for box in bndboxes if box["id"]!=aargs.sku_id]
    #for obj in bndboxes:
    #    box = obj
        #if "ignore" in box and  box["ignore"]:
        #    continue
    #    if box["id"] == aargs.sku_id:
    #        bndboxes.remove(box)
    #        dirty = True


    #if dirty:
    with open(json_path, "w", newline="\n" ) as json_file:
        json.dump(json_data, json_file, indent=4)



if __name__ == '__main__':
    input_folder = abspath(aargs.input_folder)
    # find the photo files and annotation files
    files = find_files(input_folder)
    total = len(files)

    def worker(json_path,dummy):
        remove_sku(json_path)
        return 1

    results = []
    pool = ThreadPool(12)
    counter = 0
    print("Start iterating over files. Total files: {}".format(total))
    for json_path in files:
        results.append(pool.apply_async(worker, args=(json_path,"")))

    percentage = -1
    for idx, ret in enumerate(results):
        ret.get()
        current_perc = idx * 100 // len(results)
        if current_perc != percentage:
            sys.stdout.write("\r%d%% images has been processed" % (current_perc))
            sys.stdout.flush()
            percentage = current_perc

    print("\nEnd")


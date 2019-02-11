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
                    default="/home/keyong/Downloads/anno/public/working/field_test_20181004",
                    #default="/Users/keyong/Documents/anno/public/working/RB_AD_test",
                    #default="/Users/keyong/Documents/anno/public/working/RB_Total_train_Flag",
                    help="put entire directory")
args = parser.parse_args()

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


def resize_json_image(json_data, image, json_path, photo_file):
    new_w, new_h = target_w, target_h
    (h, w, c) = image.shape
    bndboxes = json_data["bndboxes"]
    if len(bndboxes) == 0:
        pass
    else:
        xmin, xmax, ymin, ymax, area = 10000, 0, 10000, 0, []  # arbitrary
        for bndboxes in json_data["bndboxes"]:
            xmin = min(xmin, bndboxes["x"])
            ymin = min(ymin, bndboxes["y"])
            xmax = max(xmax, (bndboxes["x"] + bndboxes["w"]))
            ymax = max(ymax, (bndboxes["y"] + bndboxes["h"]))
            if xmax > w or ymax > h:
                print("There is a bndbox outside of height and width")
                print("The filename is : " + json_data["filename"])
                print("xmax: {} , width : {}".format(xmax, w))
                print("ymax: {} , height : {}".format(ymax, h))
                print("The bndbox is : {} ".format(bndboxes))
                exit()
            area.append(bndboxes["w"] * bndboxes["h"])

        area.sort()
        idx = int(len(area) * .1)
        if idx == 0:
            idx = 1

        areas_smallest_10percent = area[:idx]
        area_min = np.mean(areas_smallest_10percent)
        k_min_pixel = min(np.sqrt(float(args.min_pixel) / area_min), 1)
        new_h = h * k_min_pixel
        if new_h < target_h:
            new_h = target_h

    new_w = new_h * 1.0 / h * w
    if new_w < target_w:
        new_w = target_w
        new_h = int(new_w * 1.0 / w * h)

    k_resize = min(1.0, new_h * 1.0 / h)
    new_w = w*k_resize
    new_h = h*k_resize
    json_data["image_width"] = new_w
    json_data["image_height"] = new_h
    for i in range(len(json_data["bndboxes"])):
        json_data["bndboxes"][i]["x"] = int(json_data["bndboxes"][i]["x"] * k_resize)
        json_data["bndboxes"][i]["y"] = int(json_data["bndboxes"][i]["y"] * k_resize)
        json_data["bndboxes"][i]["w"] = int(json_data["bndboxes"][i]["w"] * k_resize)
        json_data["bndboxes"][i]["h"] = int(json_data["bndboxes"][i]["h"] * k_resize)
    image = cv2.resize(image, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)

    WRITE = True
    if WRITE:
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        cv2.imwrite(photo_file, image)


#def clean_contradictions(json_data, json_path):
def clean_conflictions(json_path):
    json_data = None

    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    dirty = False
    bndboxes = json_data.get("bndboxes")
    for obj in bndboxes:
        box = obj
        #if "ignore" in box and  box["ignore"]:
        #    continue
        if "conflict" in box:
            if box["conflict"] == True:
                print("Cannot clean up, the confliction in file(%s) is not solved!" %(json_path))
                return

            del box["conflict"]
            dirty = True


        if "conflictBox" in box:
            del box["conflictBox"]
            dirty = True

    #remove file level's contradiction flag
    if "conflict" in json_data:
        del json_data["conflict"]
        dirty = True

    if dirty:
        with open(json_path, "w", newline="\n" ) as json_file:
            json.dump(json_data, json_file, indent=4)



if __name__ == '__main__':
    input_folder = abspath(args.input_folder)
    # find the photo files and annotation files
    files = find_files(input_folder)
    total = len(files)

    def worker(json_path,dummy):
        clean_conflictions(json_path)
        return 1

    results = []
    pool = ThreadPool(12)
    counter = 0
    print("Start iterating over files. Total files: {}".format(total))
    for json_path in files:
        results.append(pool.apply_async(worker, args=(json_path,"")))

        #if counter % 20 == 0:
        #    print("{}% Completed: {}/{}".format(done, counter, total))

    percentage = -1
    for idx, ret in enumerate(results):
        ret.get()
        current_perc = idx * 100 // len(results)
        if current_perc != percentage:
            sys.stdout.write("\r%d%% images has been processed" % (current_perc))
            sys.stdout.flush()
            percentage = current_perc

    print("\nEnd")


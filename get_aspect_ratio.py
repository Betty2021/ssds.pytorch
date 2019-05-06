import argparse
import cv2
import json
from shutil import copytree
import os
import fnmatch
from os.path import abspath, join, splitext, basename
import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", type=str, help="put entire directory")
args = parser.parse_args()

DEBUG = False


def find_image_files(out_folder):
    photos_path = join(out_folder, 'photos')
    if not os.path.exists(photos_path):
        photos_path = join(out_folder, 'Photos')
    assert os.path.exists(photos_path), "photo folder is not found at {}".format(photos_path)
    all_files = os.listdir(photos_path)
    photo_files = []
    for file in all_files:
        if not file.startswith("."):
            if    fnmatch.fnmatch(file, '*.jpg') \
               or fnmatch.fnmatch(file, '*.JPG'):
                photo_files.append(os.path.join(photos_path, file))
    photo_files = sorted(photo_files)
    return photo_files, photos_path


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
    photo_files, photos_path = find_image_files(out_folder)
    annotation_files = find_annotation_files(photos_path)

    #assert len(photo_files) == len(annotation_files), "Number of images and annotations are not the same!"
    files = [(c, b) for c, b in zip(photo_files, annotation_files)]

    def check_files(photo_file, annotation_file):
        image_file = splitext(basename(annotation_file))[0]
        photo = basename(photo_file)
        assert image_file == photo, "Mismatch in image:%s and annotations:%s" % (image_file, photo)

    for (photo_file, annotation_file) in files:
        check_files(photo_file, annotation_file)
    return files


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


def add_aspect_ratio(json_data, lst):
    bndboxes = json_data.get("bndboxes")

    #bndboxes = list(filter(lambda box: "ignore" not in box or not box["ignore"], bndboxes))
    bndboxes = list(filter(lambda box: ("ignore" not in box or not box["ignore"]) and "competitor" not in box["id"], bndboxes))
    for obj in bndboxes:
        box = obj
        w = float(box["w"])
        h = float(box["h"])
        x = float(box["x"])
        y = float(box["y"])
        #ratio = h/w
        ratio = w/h
        lst.append((ratio,1.0))

   # return image


if __name__ == '__main__':
    input_folder = abspath(args.input_folder)

    # find the photo files and annotation files
    files = find_files(input_folder)
    total = len(files)
    counter = 0
    ratio_list=[]
    print("Start iterating over files. Total files: {}".format(total))
    for (photo_path, json_path) in files:
        #try:
        #    image = cv2.imread(photo_path)
        #except:
        #    print("can't open jpg file:%s" % (photo_path))

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        add_aspect_ratio(json_data,ratio_list )

        counter += 1
        progress = counter*1.0 / total
        done = int(progress * 100)
        #if counter % 20 == 0:
        #    print("{}% Completed: {}/{}".format(done, counter, total))
    #kmeans=KMeans(n_clusters=6, random_state=0).fit(ratio_list)
    kmeans=KMeans(n_clusters=5, random_state=0).fit(ratio_list)
    print(kmeans.cluster_centers_)
    aspect_ratios=[ x[0] for x in kmeans.cluster_centers_]
    #print("kmeans: ["+str(kmeans.cluster_centers_[0][0])+","+ str(kmeans.cluster_centers_[1][0])+","+str(kmeans.cluster_centers_[2][0])+"]")
    print("kmeans: "+str(aspect_ratios))
    #print("End")


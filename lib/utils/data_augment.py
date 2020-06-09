"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import torch
import cv2
import numpy as np
import random
import math
import imgaug.augmenters as iaa
from lib.utils.box_utils import matrix_iou
#_FOR_PMI_UKRAINE=True
_FOR_PMI_UKRAINE=False
def _crop(self, image, boxes, labels):
    height, width, _ = image.shape
    min_cropped_ratio=0.4 if _FOR_PMI_UKRAINE else 0.70
    if len(boxes)== 0 or (len(boxes)==1 and labels[0]==0):
        scale = random.uniform(0.70, 1.)
        #just keep aspect ratio
        #ratio = 1.0
        scale = random.uniform(min_cropped_ratio, 1.)
        min_ratio = max(0.8, scale*scale)
        max_ratio = min(1/0.8, 1. / scale / scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        w = int(scale * ratio * width)
        h = int((scale / ratio) * height)
        l = random.randrange(width - w)
        t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        return image_t, boxes, labels

    area=np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)/(height*width)
    big_sku = np.mean(area)>=0.05
    if big_sku:
        return image, boxes, labels
    while True:
        mode = random.choice((
            None,
            (0.001,0.05), #0.025*0.025=0.000625
            (0.002,0.10),
            (0.005,0.10),
            (0.01, 0.1),
            (0.05, 0.25),
            (0.1,  0.30),
            (0.25, 0.5),
            #(0.5, None),
            #(0.7, None),
            #(0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            #scale = random.uniform(0.50, 1.)
            #it is very strange, pmi ukraine has stock counter function,
            #the sku in such image is quite big, so need to crop out a small portion
            #and resize it to [800,600] to make a fake big sku in training stage.
            scale = random.uniform(min_cropped_ratio, 1.)
            min_ratio = max(0.85, scale*scale)
            max_ratio = min(1/0.85, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            #just keep aspect ratio
            #ratio = 1.0
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            #centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            #mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            mask = (iou>(1/2400)).squeeze(1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue
            old_boxes_t = boxes_t.copy()
            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            #
            new_area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            old_area = (old_boxes_t[:, 2] - old_boxes_t[:, 0]) * (old_boxes_t[:, 3] - old_boxes_t[:, 1])
            iou = new_area/ old_area
            mask= iou <=0.65
            bad_mask = mask
            #print(self.ambigous_skus)
            #if True:
            if len(self.ambigous_skus) > 0:
                mask_ambigous_skus_idx = [sku in self.ambigous_skus for sku in labels_t]
                #mask2 = 2<=labels_t
                mask_ambigous_skus_crop_ratio = iou < (1-self.ambigous_skus_crop_ratio)
                mask_ambigous_skus = mask_ambigous_skus_idx & mask_ambigous_skus_crop_ratio
                bad_mask = mask|mask_ambigous_skus
                pp_boxes = boxes_t[mask_ambigous_skus]
                bad_boxes_t=boxes_t[bad_mask]
                if len(pp_boxes)>1000:
                    print(self.ambigous_skus)
                    print(self.ambigous_skus_crop_ratio)
                    print(iou)
                    print(labels_t)
                    print(mask_ambigous_skus)
                    print(bad_mask)

            bad_boxes_t=boxes_t[bad_mask]
            for box in bad_boxes_t:
                #print("black out the box because iou <0.7")
                image_t[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = random.randint(0,255)

            boxes_t = boxes_t[~bad_mask]
            labels_t = labels_t [~bad_mask]
            if len(boxes_t)==0:
                targets = np.zeros((1, 5))
                boxes_t = targets[:, :-1].copy()
                labels_t = targets[:, -1].copy()

            return image_t, boxes_t,labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp = np.uint8(np.clip(tmp,0,255))
        #tmp[tmp < 0] = 0
        #tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()
    need_convert=False
    beta=0
    alpha=1
    if random.randrange(2):
        beta=random.uniform(-10, 20)
        need_convert=True

    if random.randrange(2):
        alpha=random.uniform(0.8, 1.3)
        need_convert=True

    if need_convert:
       _convert(image, alpha=alpha, beta=beta)

    #dont change hue
    #if random.randrange(2):
    #    tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
    #    tmp %= 180
    #    image[:, :, 0] = tmp

    # change saturation
    if random.randrange(2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _convert(image[:, :, 1], alpha=random.uniform(0.8, 1.3))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes
    b_w = (boxes[:, 2] - boxes[:, 0])*1.
    b_h = (boxes[:, 3] - boxes[:, 1])*1.

    min_area=np.min(b_w * b_h)/(image.shape[0]*image.shape[1])
    #if min_area< 1/40.0*1/50: #too small
    #    return image, boxes
    max_expand_ratio=2.0 if _FOR_PMI_UKRAINE else 1.4

    max_pad_scale=np.clip(math.sqrt(min_area)/(1/36.0),1.0, max_expand_ratio)
    if max_pad_scale <=1.0:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1.0, max_pad_scale)

        min_ratio = max(0.75, 1./scale/scale)
        max_ratio = min(1/0.75, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)


        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes



def _elastic(image, p, alpha=None, sigma=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     From: 
     https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    """
    if random.random() > p:
        return image
    if alpha == None:
        alpha = image.shape[0] * random.uniform(0.5,2)
    if sigma == None:
        sigma = int(image.shape[0] * random.uniform(0.5,1))
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]
    
    dx, dy = [cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1) * alpha, (sigma|1, sigma|1), 0) for _ in range(2)]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = np.clip(x+dx, 0, shape[1]-1).astype(np.float32), np.clip(y+dy, 0, shape[0]-1).astype(np.float32)
    return cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT)


def preproc_for_test(image, w_h_insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (w_h_insize[0], w_h_insize[1]),interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)

def _preproc_resize(image, w_h_insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (w_h_insize[0], w_h_insize[1]),interpolation=interp_method)
    return image

def _preproc_for_test_with_resized_img(resized_image, mean):
    image = resized_image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)

def draw_bbox(image, bbxs, color=(0, 255, 0)):
    img = image.copy()
    #img = img[...,::-1]
    bbxs = np.array(bbxs).astype(np.int32)
    for bbx in bbxs:
        cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), color, 5)
    img = img[..., ::-1]
    return img

def rotation(src_img_size , src_img, src_box_info_old):
    tar_box_info_final = src_box_info_old.copy()
    tar_img = src_img.copy()
    src_height_old = int(src_img_size[0])
    src_width_old = int(src_img_size[1])
    a = random.randint(1 , 5)
    ii = 0
    angle_pi = a * math.pi / 180.0
    crop_center = (0 , 0)
    M = cv2.getRotationMatrix2D(crop_center, a, 1.0)

    right = int(src_height_old * math.sin(angle_pi))
    top = int(src_width_old * math.sin(angle_pi))
    src_width = int(right + src_width_old)
    src_height = int(top + src_height_old)
    img_new = cv2.copyMakeBorder(src_img,top,0,0,right,cv2.BORDER_CONSTANT,value=[0,0,0])

    tar_img = cv2.warpAffine(img_new, M,(src_width, src_height))

    tar_box_info = np.zeros((2, 2))
    tar_box_info1 = np.zeros((2, 2))
    tar_box_info2 = np.zeros((2, 2))
    for z,item in enumerate(src_box_info_old):
        src_box_info = item
        src_box_info1 = [src_box_info[0], (((src_box_info[3] - src_box_info[1]) / 2) + src_box_info[1] + top), src_box_info[2],
                         ((src_box_info[3] - src_box_info[1]) / 2) + src_box_info[1] + top]
        src_box_info2 = [((src_box_info[2] - src_box_info[0]) / 2) + src_box_info[0], src_box_info[1] + top,
                         ((src_box_info[2] - src_box_info[0]) / 2) + src_box_info[0], src_box_info[3] + top]

        for i in range(2):
            tar_box_info1[i][0] = ((src_box_info1[2 * i]) * math.cos(angle_pi) + (src_box_info1[2 * i + 1]) * math.sin(angle_pi))
            tar_box_info2[i][1] = ((src_box_info2[2 * i + 1]) * math.cos(angle_pi) - (src_box_info2[2 * i]) * math.sin(angle_pi))
            if tar_box_info1[i][0] < 0:
                tar_box_info1[i][0] = 0
            elif tar_box_info1[i][0] > src_width:
                tar_box_info1[i][0] = src_width
            if tar_box_info2[i][1] < 0:
                tar_box_info2[i][1] = 0
            elif tar_box_info2[i][1] > src_height:
                tar_box_info2[i][1] = src_height

        tar_box_info = tar_box_info1[0][0],tar_box_info2[0][1],tar_box_info1[1][0],tar_box_info2[1][1]
        tar_box_info_final[z]=tar_box_info
        ii+=1
    return tar_img , tar_box_info_final[0:ii]

class preproc(object):

    def __init__(self, resize, rgb_means, p, writer=None, ambigous_skus=[],ambigous_skus_crop_ratio=0.35):
        self.means = rgb_means
        self.w_h_resize = [resize[1],resize[0]]  #opencv's resize, which is w,h
        self.p = p
        self.writer = writer # writer used for tensorboard visualization
        self.epoch = 0
        self.ambigous_skus = ambigous_skus
        self.ambigous_skus_crop_ratio = ambigous_skus_crop_ratio
        if p>=0 and p <=1:
            sometimes = lambda aug: iaa.Sometimes(p, aug)
            self.seq = iaa.Sequential(
                [
                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0, 1.0)),
                        iaa.AverageBlur(k=(2, 5)),
                        #iaa.MedianBlur(k=(2, 4)),
                    ]),
                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                    #sometimes(iaa.Affine(rotate=(1, 5))),
                    # Add gaussian noise to some images.
                    #sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                    sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.5)),
                    # Add a value of -5 to 5 to each pixel.
                    #sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                    # Change brightness of images (80-120% of original value).
                    #sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                    # Improve or worsen the contrast of images.
                    sometimes(iaa.ContrastNormalization((0.8, 1.2))),
                ],
                # do all of the above augmentations in random order
                random_order=True
            )

    def __call__(self, image, targets=None):
        # some bugs 
        if self.p == -2: # abs_test
            targets = np.zeros((1,5))
            #targets[0] = image.shape[0]
            #targets[0] = image.shape[1]
            image = preproc_for_test(image, self.w_h_resize, self.means)
            return torch.from_numpy(image), targets

        #print(targets)
        boxes = targets[:,:-1].copy()
        labels = targets[:,-1].copy()
        
        if len(boxes) == 0:
            targets = np.zeros((1,5))
            boxes = targets[:,:-1].copy()
            labels = targets[:,-1].copy()
             #image = preproc_for_test(image, self.w_h_resize, self.means) # some ground truth in coco do not have bounding box! weird!
            #return torch.from_numpy(image), targets
        if self.p == -1: # eval
            height, width, _ = image.shape
            boxes[:, 0::2] /= width
            boxes[:, 1::2] /= height
            labels = np.expand_dims(labels,1)
            targets = np.hstack((boxes,labels))
            image = preproc_for_test(image, self.w_h_resize, self.means)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-1]
        labels_o = targets_o[:,-1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        targets_o = np.hstack((boxes_o,labels_o))

        if self.writer is not None:
            image_show = draw_bbox(image, boxes)
            self.writer.add_image('preprocess/input_image', image_show, self.epoch, dataformats='HWC')
        ran = random.randint(0 , 1)
        if ran == 1:
            image, boxes = rotation(image.shape, image, boxes)

        image_t, boxes, labels = _crop(self, image, boxes, labels)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/crop_image', image_show, self.epoch,dataformats='HWC')

        distorted = False
        h, w ,_ = image_t.shape
        if h*w < self.w_h_resize[0]* self.w_h_resize[1]:
            distorted = True
            image_t = _distort(image_t)
            if self.writer is not None:
                image_show = draw_bbox(image_t, boxes)
                self.writer.add_image('preprocess/distort_image', image_show, self.epoch, dataformats='HWC')
        
        # image_t = _elastic(image_t, self.p)
        # if self.writer is not None:
        #     image_show = draw_bbox(image_t, boxes)
        #     self.writer.add_image('preprocess/elastic_image', image_show, self.epoch)

        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/expand_image', image_show, self.epoch, dataformats='HWC')

        #image_t, boxes = _mirror(image_t, boxes)
        #if self.writer is not None:
        #    image_show = draw_bbox(image_t, boxes)
        #    self.writer.add_image('preprocess/mirror_image', image_show, self.epoch)


        height, width, _ = image_t.shape
        image_t = _preproc_resize(image_t, self.w_h_resize)
        if not distorted:
            image_t = _distort(image_t)
            if self.writer is not None:
                boxes1 = boxes.copy()
                boxes1[:, 0::2] *= self.w_h_resize[0]/width
                boxes1[:, 1::2] *= self.w_h_resize[1]/height
                image_show = draw_bbox(image_t, boxes1)
                self.writer.add_image('preprocess/distort_image', image_show, self.epoch, dataformats='HWC')

        seq_det = self.seq.to_deterministic()
        image_t = seq_det.augment_images([image_t])[0]
        if self.writer is not None:
            boxes1 = boxes.copy()
            boxes1[:, 0::2] *= self.w_h_resize[0]/width
            boxes1[:, 1::2] *= self.w_h_resize[1]/height
            image_show = draw_bbox(image_t, boxes1)
            self.writer.add_image('preprocess/iaa_image', image_show, self.epoch, dataformats='HWC')

        # only write the preprocess step for the first image
        if self.writer is not None:
            # print('image adding')
            self.release_writer()

        image_t = _preproc_for_test_with_resized_img(image_t, self.means)

        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        #mask_b= np.minimum(b_w, b_h) > 0.015
        area= b_w*b_h
        #if len(boxes)==0:
        #    x=0
        #    x+=1
        mask_b= np.logical_and(area> 0.0003, area < 0.15)
        #mask_b= (b_w*b_h) > 0.0003  #area should
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        #print('image_t:  '+ str(image_t))
        #image_t, boxes_t = rotation(image_t.shape, image_t, boxes_t)

        if len(boxes_t)==0:
            image = preproc_for_test(image_o, self.w_h_resize, self.means)
            return torch.from_numpy(image),targets_o

        labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))

        return torch.from_numpy(image_t), targets_t

    # def _imgaug_(self, image, targets=None):
    #     if self.p == -2: # abs_test
    #         targets = np.zeros((1,5))
    #         image = preproc_for_test(image, self.w_h_resize, self.means)
    #         return torch.from_numpy(image), targets
    #
    #     boxes = targets[:, :-1].copy()
    #     labels = targets[:, -1].copy()
    #
    #     if self.p == -1:  # eval
    #         height, width, _ = image.shape
    #         boxes[:, 0::2] /= width
    #         boxes[:, 1::2] /= height
    #         labels = np.expand_dims(labels, 1)
    #         targets = np.hstack((boxes, labels))
    #         image = preproc_for_test(image, self.w_h_resize, self.means)
    #         return torch.from_numpy(image), targets




    def add_writer(self, writer, epoch=None):
        self.writer = writer
        self.epoch = epoch if epoch is not None else self.epoch + 1
    
    def release_writer(self):
        self.writer = None

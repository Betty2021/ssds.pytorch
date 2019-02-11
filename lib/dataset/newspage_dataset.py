import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
from .voc_eval import voc_ap


class NPSet(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,  image_sets=None, preproc=None):
        self.root = root
        self.preproc = preproc
        self.name, self.name_to_seq, self.seq_to_name, self.name_to_desc= self._parse_templates()
        self.ids, self.photo_dir, self.anno_dir = self._find_image_annotation_pair()
        #print("id of 181004142415 is ", self.ids.index("OEX_181004142415.jpg") )
        if image_sets and isinstance(image_sets,list) and isinstance(image_sets[0],int) :
            self.ids=self.ids*image_sets[0]
        self.num_classes=len(self.seq_to_name)
        #self.images, self.targets=self.read_all_images()

    def _parse_templates(self):
        templates_path= os.path.join(self.root, "templates.json")
        with open(templates_path, "r") as json_file:
            json_data = json.load(json_file)
        skus=json_data["categories"][0]["skus"]
        skus.insert(0,{"id":"background","name":"background"})
        cls_dict = { sku["id"]: idx for idx, sku in enumerate(skus)}
        cls_to_desc_dict = { sku["id"]: sku["name"] for sku in skus}
        cls_seqno_to_id  = [ sku["id"]  for sku in skus]
        dataset_name=json_data["name"]
        return dataset_name, cls_dict, cls_seqno_to_id, cls_to_desc_dict


    def _find_image_annotation_pair(self):
        photos_path = os.path.join(self.root, 'photos')
        if not os.path.exists(photos_path):
            photos_path = os.path.join(self.root, 'Photos')
        assert os.path.exists(photos_path), "photo folder is not found at {}".format(self.root)
        anno_path = os.path.join(photos_path, 'Annotations')
        assert os.path.exists(anno_path), "Annotation folder is not found at {}".format(anno_path)

        all_files = os.listdir(photos_path)
        photo_files = []
        for file in all_files:
            up_file=file.upper()
            if not up_file.startswith("."):
                if up_file.endswith('.JPG') or up_file.endswith('.JPEG'):
                    if os.path.exists(os.path.join(anno_path,file+".json")):
                        photo_files.append(file)

        return photo_files, photos_path, anno_path

    def _to_target(self, bndboxes):
        """return nparray, shape(n, 5)
                             [
                              [xmin, ymin,xmax, ymax, label],
                              [xmin, ymin,xmax, ymax, label]
                              ]
        """
        res = np.empty((0, 5))
        for obj in bndboxes:
            if "ignore" in obj.keys() and obj["ignore"]:
                continue
            name = obj['id']
            bndbox = [
                      int(obj["x"]),
                      int(obj["y"]),
                      int(obj["w"]),
                      int(obj["h"])
            ]
            bndbox[2]=bndbox[0]+bndbox[2]
            bndbox[3]=bndbox[1]+bndbox[3]
            if name in self.name_to_seq.keys():
                label_idx = self.name_to_seq[name]
                bndbox.append(label_idx)
                res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res

    def read_all_images(self):
        images = list()
        targets = list()
        for index in range(len(self.ids)):
            img_id = self.ids[index]
            with open(os.path.join(self.anno_dir,img_id+".json"),"r") as json_file:
                json_data = json.load(json_file)
            bndboxes=json_data["bndboxes"]
            target = self._to_target(bndboxes)
            img = cv2.imread(os.path.join(self.photo_dir, img_id), cv2.IMREAD_COLOR)
            images.append(img)
            targets.append(target)
        return images, targets

    def __getitem__(self, index):
        # obj=self._cache[index]
        # if obj:
        #    img, target = obj
        # else:
        #     if index ==10:
        #         x=10
        img_id = self.ids[index]

        with open(os.path.join(self.anno_dir,img_id+".json"),"r") as json_file:
            json_data = json.load(json_file)
        bndboxes=json_data["bndboxes"]
        target = self._to_target(bndboxes)
        img = cv2.imread(os.path.join(self.photo_dir, img_id), cv2.IMREAD_COLOR)
        #target= self.targets[index]
        height, width, _ = img.shape


        if self.preproc is not None:
            img, target = self.preproc(img, target)
            #print(img.shape)

        # print(target.shape)
        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(os.path.join(self.photo_dir, img_id), cv2.IMREAD_COLOR)



    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]

        with open(os.path.join(self.anno_dir, img_id + ".json"), "r") as json_file:
            json_data = json.load(json_file)
        bndboxes = json_data["bndboxes"]
        anno = self._to_target(bndboxes)
        return anno
    #
    # def pull_img_anno(self, index):
    #     '''Returns the original annotation of image at index
    #
    #     Note: not using self.__getitem__(), as any transformations passed in
    #     could mess up this functionality.
    #
    #     Argument:
    #         index (int): index of img to get annotation of
    #     Return:
    #         list:  [img_id, [(label, bbox coords),...]]
    #             eg: ('001718', [('dog', (96, 13, 438, 332))])
    #     '''
    #     img_id = self.ids[index]
    #     img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
    #     anno = ET.parse(self._annopath % img_id).getroot()
    #     gt = self.target_transform(anno)
    #     height, width, _ = img.shape
    #     boxes = gt[:, :-1]
    #     labels = gt[:, -1]
    #     boxes[:, 0::2] /= width
    #     boxes[:, 1::2] /= height
    #     labels = np.expand_dims(labels, 1)
    #     targets = np.hstack((boxes, labels))
    #
    #     return img, targets
    #
    # def pull_tensor(self, index):
    #     '''Returns the original image at an index in tensor form
    #
    #     Note: not using self.__getitem__(), as any transformations passed in
    #     could mess up this functionality.
    #
    #     Argument:
    #         index (int): index of img to show
    #     Return:
    #         tensorized version of img, squeezed
    #     '''
    #     to_tensor = transforms.ToTensor()
    #     return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, dontcare):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)


        aps, map = self.do_python_eval()
        return aps, map

    def _get_np_results_folder(self):
        filedir=  os.path.join( self.root, 'results', self.name, 'np_result')
        if not os.path.exists(filedir):
           os.makedirs(filedir)
        return filedir

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            self.root, 'results', self.name, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.seq_to_name):
            #cls_ind = cls_ind
            if cls == 'background' or cls_ind == 0:
                continue
            #print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    #index = index
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


    def np_eval(self, detpath, classname, ground_trues, ovthresh=0.5, use_07_metric=False):
       # extract gt objects for this class
        class_recs = {}
        npos = 0

        for imagename in self.ids:
            R = [obj for obj in ground_trues[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}


        # read dets
        #detfile = detpath.format(classname)
        detfile = detpath
        with open(detfile, 'r') as f:
            lines = f.readlines()

        # splitlines = [x.strip().split(' ') for x in lines]
        # image_ids = [x[0] for x in splitlines]
        # confidence = np.array([float(x[1]) for x in splitlines])
        # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        splitlines = [x.strip().split(' ') for x in lines]
        confidence = np.array([float(x[-5]) for x in splitlines])
        BB = np.array([[float(z) for z in x[-4:]] for x in splitlines])
        image_ids = [" ".join(x[:-5]) for x in splitlines]
        if BB.shape[0]==0:
            return 1.0, 1.0, 1.0
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

            # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap =voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap


    def save_np_result(self):
        det={}
        for i, cls in enumerate(self.seq_to_name):
            if i == 0:
                continue

            det_filename = self._get_voc_results_file_template().format(cls)
            # read dets
            #detfile = det_filename.format(det_filename)
            with open(det_filename, 'r') as f:
                lines = f.readlines()

            # splitlines = [x.strip().split(' ') for x in lines]
            # image_ids = [x[0] for x in splitlines]
            # confidence = np.array([float(x[1]) for x in splitlines])
            # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            splitlines = [x.strip().split(' ') for x in lines]
            confidence = np.array([float(x[-5]) for x in splitlines])
            BB = [[float(z) for z in x[-4:]] for x in splitlines]
            image_ids = [" ".join(x[:-5]) for x in splitlines]
            for row, image_id in enumerate(image_ids):
                if confidence[row]<0.4:
                    continue

                if image_id not in det.keys():
                   det[image_id]={cls:[]}
                if cls not in det[image_id].keys():
                    det[image_id][cls]=[]

                det[image_id][cls]=det[image_id][cls]+ [BB[row]]

        for file_name in self.ids:
            json_file=os.path.join(self._get_np_results_folder(),file_name+".json")
            with open(json_file, "w") as output_fp:
                output_fp.write("""{
                "version": "1.0.0",
                "company": "RB_Test",
                "dataset": "photos",
                "filename": "%s",
                "image_width": %d,
                "image_height": %d,
                "bndboxes": [
            """ % (file_name, 10, 10))
                str = ""
                if file_name in det.keys():
                    det_of_file = det[file_name]
                    for cls in det_of_file.keys():
                        boxes=det_of_file[cls]
                        for box in boxes:
                            xmin, ymin, xmax, ymax = box
                            str += """
                            {
                                "x": %f,
                                "y": %f,
                                "w": %f,
                                "h": %f,
                                "id": "%s",
                                "strokeStyle": "#3399FF",
                                "fillStyle": "#00FF00"
                            },""" % (xmin , ymin , (xmax - xmin) , (ymax - ymin) , cls)

                    if len(str) > 0:
                        str = str[:-1]

                output_fp.write(str)
                output_fp.write("\n]}")


    def do_python_eval(self):
        ground_trues = {}   # {"image_oo1":[{name:sku_x,bbox:[xmin,ymin,xmax,ymax]},....], "image_002"}
        self.save_np_result()
        for i, img_id in enumerate(self.ids):
            with open(os.path.join(self.anno_dir, img_id + ".json"), "r") as json_file:
                json_data = json.load(json_file)
            bndboxes = json_data["bndboxes"]
            objects = []
            for obj in bndboxes:
                name = obj['id']
                if "ignore" in obj.keys() and obj["ignore"]:
                    continue
                if name in self.name_to_seq.keys():
                    obj_struct = {}
                    obj_struct['name'] = name

                    bndbox = [
                        int(obj["x"]),
                        int(obj["y"]),
                        int(obj["w"]),
                        int(obj["h"])
                    ]
                    bndbox[2] = bndbox[0] + bndbox[2]
                    bndbox[3] = bndbox[1] + bndbox[3]
                    obj_struct['bbox'] = bndbox
                    obj_struct['difficult'] = 0
                    objects.append(obj_struct)

            ground_trues[img_id] = objects

        aps = []
        use_07_metric = True
        #use_07_metric = False
        #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        for i, cls in enumerate(self.seq_to_name):
            if cls == 'background':
                continue

            det_filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = self.np_eval(
                det_filename, cls, ground_trues, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(self.name_to_desc[cls], ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')
        return aps, np.mean(aps)

    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255, 0, 0), 3)
        cv2.imwrite('./image.jpg', img)

# ## test
# if __name__ == '__main__':
#     ds = NPSet('/home/keyong/Documents/ssd/rb_harpic.git_resized')
#     print(len(ds))
#     ds.do_python_eval()
#
#     #img, target = ds[0]
#     #print(target)
#     #ds.show(1)
#     exit(0)
from dataset_base import BaseDataset
import os
import numpy as np
import cv2
import glob



class Kaggle(BaseDataset):
    def __init__(self, data_dir, phase, transform=None):
        super(Kaggle, self).__init__(data_dir, phase, transform)
        self.class_name = ['__background__', 'kaggle']
        self.num_classes = len(self.class_name)-1

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.img_dir, img_id, "images", img_id+'.png')
        img = cv2.imread(imgFile)
        return img

    def load_gt_masks(self, annopath):
        masks = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*" + '.png'))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                masks.append(np.where(mask > 0, 1., 0.))
        return np.asarray(masks, np.float32)


    def load_gt_bboxes(self, annopath):
        bboxes = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*" + '.png'))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([y1, x1, y2, x2])
        return np.asarray(bboxes, np.float32)

    def load_annoFolder(self, index):
        img_id = self.img_ids[index]
        return os.path.join(self.img_dir, img_id, "masks")

    def load_annotation(self, index, type='mask'):
        annoFolder = self.load_annoFolder(index)
        if type=='mask':
            return self.load_gt_masks(annoFolder)
        else:
            return self.load_gt_bboxes(annoFolder)

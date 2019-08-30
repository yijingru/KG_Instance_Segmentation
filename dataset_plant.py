from dataset_base import BaseDataset
import os
import numpy as np
import cv2

class Plant(BaseDataset):
    def __init__(self, data_dir, phase, transform=None):
        super(Plant, self).__init__(data_dir, phase, transform)
        self.class_name = ['__background__', 'plant']
        self.num_classes = len(self.class_name)-1

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.img_dir, img_id)
        img = cv2.imread(imgFile)
        return img

    def load_gt_masks(self, annopath):
        masks = []
        mask_gt = cv2.imread(annopath)
        h, w, _ = mask_gt.shape
        cond1 = mask_gt[:, :, 0] != mask_gt[:, :, 1]
        cond2 = mask_gt[:, :, 1] != mask_gt[:, :, 2]
        cond3 = mask_gt[:, :, 2] != mask_gt[:, :, 0]

        r, c = np.where(cond1+cond2+cond3)
        unique_colors = np.unique(mask_gt[r, c, :], axis=0)

        for color in unique_colors:
            cond1 = mask_gt[:, :, 0] == color[0]
            cond2 = mask_gt[:, :, 1] == color[1]
            cond3 = mask_gt[:, :, 2] == color[2]
            r, c = np.where(cond1*cond2*cond3)
            y1 = np.min(r)
            x1 = np.min(c)
            y2 = np.max(r)
            x2 = np.max(c)
            if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                continue
            masks.append(np.where(cond1*cond2*cond3, 1., 0.))
        return np.asarray(masks, np.float32)


    def load_gt_bboxes(self, annopath):
        bboxes = []
        masks = self.load_gt_masks(annopath)
        for mask in masks:
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

    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'masks', img_id)

    def load_annotation(self, index, type='mask'):
        img_id = self.img_ids[index]
        annoFolder = self.load_annoFolder(img_id)
        if type=='mask':
            return self.load_gt_masks(annoFolder)
        else:
            return self.load_gt_bboxes(annoFolder)

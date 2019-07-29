from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import torch
import preprocessing
import config as cfg
import glob



class Kaggle(Dataset):
    def __init__(self, data_dir, phase, transform=None):
        super(Kaggle, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_dir = os.path.join(data_dir, phase)
        self.img_ids = sorted(os.listdir(self.img_dir))
        self.class_name = ['__background__', 'cell']
        self.num_classes = len(self.class_name)-1

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.img_dir, img_id, "images", img_id+'.png')
        img = cv2.imread(imgFile)
        return img

    def load_gt_masks(self, annopath, suffix):
        masks = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*" + suffix))):
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


    def load_gt_bboxes(self, annopath, suffix):
        bboxes = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*" + suffix))):
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
            return self.load_gt_masks(annoFolder, '.png')
        else:
            return self.load_gt_bboxes(annoFolder, 'png')


    def transfer_bboxes(self, bboxes_c0):
        out_box = []
        for i in range(bboxes_c0.shape[0]):
            tl, tr, bl, br, cc = bboxes_c0[i,:,:]
            y1 = tl[1]
            x1 = tl[0]
            y2 = br[1]
            x2 = br[0]
            out_box.append([y1,x1,y2,x2,1])
        return np.asarray(out_box, np.float32)

    def load_gt_masks_bboxes(self, instance_masks):
        gt_masks = []
        gt_bboxes = []
        for mask in instance_masks:
            rr,cc = np.where(mask==1.)
            if len(rr>1):
                y1 = np.min(rr)
                x1 = np.min(cc)
                y2 = np.max(rr)
                x2 = np.max(cc)
                if abs(y2-y1)>2 and abs(x2-x1)>2:
                    gt_masks.append(mask)
                    gt_bboxes.append([y1,x1,y2,x2,1])
        return np.asarray(gt_masks, np.float32), np.asarray(gt_bboxes, np.float32)

    def masks_to_bboxes(self, masks, divide_scale=1.):
        bboxes = []
        height, width = masks[0].shape
        height_scale = int(float(height) / float(divide_scale))
        width_scale = int(float(width) / float(divide_scale))

        for mask in masks:
            mask_scale = cv2.resize(mask, (width_scale, height_scale), interpolation=cv2.INTER_NEAREST)
            rr, cc = np.where(mask_scale == 1.)
            if len(rr > 1):
                y1 = np.min(rr)
                x1 = np.min(cc)
                y2 = np.max(rr)
                x2 = np.max(cc)
                if ((y2 - y1) > cfg.KP_RADIUS * 2 + 1 and (x2 - x1) > cfg.KP_RADIUS * 2 + 1):
                    tl = (x1, y1)
                    tr = (x2, y1)
                    bl = (x1, y2)
                    br = (x2, y2)
                    cc = (float(x1 + x2) / 2, float(y1 + y2) / 2)
                    bboxes.append([tl, tr, bl, br, cc])
        return np.asarray(bboxes, np.float32), height_scale, width_scale

    def __getitem__(self, item):
        img = self.load_image(item)
        height,width,_ = img.shape
        instance_masks = self.load_annotation(item, type='mask')

        if self.transform is not None:
            img, instance_masks = self.transform(img, instance_masks)

        bboxes_c0, h_c0, w_c0 = self.masks_to_bboxes(instance_masks, divide_scale = 1.)
        bboxes_c1, h_c1, w_c1 = self.masks_to_bboxes(instance_masks, divide_scale = 2.)
        bboxes_c2, h_c2, w_c2 = self.masks_to_bboxes(instance_masks, divide_scale = 4.)
        bboxes_c3, h_c3, w_c3 = self.masks_to_bboxes(instance_masks, divide_scale = 8.)

        kp0, short0, mid0 = preprocessing.get_ground_truth(bboxes_c0, h_c0, w_c0, num_kps=cfg.NUM_KPS)
        kp1, short1, mid1 = preprocessing.get_ground_truth(bboxes_c1, h_c1, w_c1, num_kps=cfg.NUM_KPS)
        kp2, short2, mid2 = preprocessing.get_ground_truth(bboxes_c2, h_c2, w_c2, num_kps=cfg.NUM_KPS)
        kp3, short3, mid3 = preprocessing.get_ground_truth(bboxes_c3, h_c3, w_c3, num_kps=cfg.NUM_KPS)

        gt_c0 = np.concatenate((kp0, np.transpose(short0, (2,0,1)), np.transpose(mid0, (2,0,1))), 0)
        gt_c1 = np.concatenate((kp1, np.transpose(short1, (2,0,1)), np.transpose(mid1, (2,0,1))), 0)
        gt_c2 = np.concatenate((kp2, np.transpose(short2, (2,0,1)), np.transpose(mid2, (2,0,1))), 0)
        gt_c3 = np.concatenate((kp3, np.transpose(short3, (2,0,1)), np.transpose(mid3, (2,0,1))), 0)

        img = np.clip(img, a_min=0., a_max=255.)
        img = np.float32(img)/255-0.5
        img = np.transpose(img, (2,0,1))

        img = torch.FloatTensor(img)
        gt_c0 = torch.FloatTensor(gt_c0)
        gt_c1 = torch.FloatTensor(gt_c1)
        gt_c2 = torch.FloatTensor(gt_c2)
        gt_c3 = torch.FloatTensor(gt_c3)

        gt_masks, gt_bboxes = self.load_gt_masks_bboxes(instance_masks)

        return img, gt_c0, gt_c1, gt_c2, gt_c3, gt_masks, gt_bboxes

    def __len__(self):
        return len(self.img_ids)

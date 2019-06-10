from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import torch
import preprocessing
import config as cfg
import glob


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)

def load_gt_bboxes(annopath, suffix):
    bboxes = []
    for annoImg in sorted(glob.glob(os.path.join(annopath, "*"+suffix))):
        mask = cv2.imread(annoImg, -1)
        r,c = np.where(mask>0)
        if len(r):
            y1 = np.min(r)
            x1 = np.min(c)
            y2 = np.max(r)
            x2 = np.max(c)
            if (abs(y2-y1)<=1 or abs(x2-x1)<=1):
                continue
            bboxes.append([y1,x1,y2,x2])
    return np.asarray(bboxes, np.float32)

def load_bboxes(masks, divide_scale = 1., maxi = 5.):
    bboxes = []
    height, width = masks[0].shape
    height_scale = int(float(height)/float(divide_scale))
    width_scale  = int(float(width)/float(divide_scale))

    for mask in masks:
        mask_scale = cv2.resize(mask, (width_scale, height_scale), interpolation=cv2.INTER_NEAREST)
        rr,cc = np.where(mask_scale==1.)
        if len(rr>1):
            y1 = np.min(rr)
            x1 = np.min(cc)
            y2 = np.max(rr)
            x2 = np.max(cc)
            if ((y2-y1)>cfg.KP_RADIUS*2+1 and (x2-x1)>cfg.KP_RADIUS*2+1):# and ((y2-y1)<cfg.KP_RADIUS*2*maxi) and ((x2-x1)<cfg.KP_RADIUS*2*maxi):
                tl = (x1,y1)
                tr = (x2,y1)
                bl = (x1,y2)
                br = (x2,y2)
                cc = (float(x1+x2)/2, float(y1+y2)/2)
                bboxes.append([tl,tr,bl,br,cc])
    return np.asarray(bboxes, np.float32), height_scale, width_scale


def load_gt_masks(annopath, suffix):
    masks = []
    for annoImg in sorted(glob.glob(os.path.join(annopath, "*"+suffix))):
        mask = cv2.imread(annoImg, -1)
        mask = np.where(mask>0, 1., 0.)
        if mask.sum()>0.:
            masks.append(mask)
    return np.asarray(masks)


def load_gt_masks_resize(annopath, suffix, height, width):
    masks = []
    for annoImg in sorted(glob.glob(os.path.join(annopath, "*"+suffix))):
        mask = cv2.imread(annoImg, -1)
        mask = np.where(mask>0, 1., 0.)
        if mask.sum()>0.:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
    return np.asarray(masks)

class NeucleiDataset(Dataset):
    def __init__(self, dataDir, annoDir, dataSuffix, annoSuffix, transform=None):
        super(NeucleiDataset, self).__init__()
        self.dataDir = dataDir
        self.dataSuffix = dataSuffix
        self.annoSuffix = annoSuffix
        self.transform = transform
        self.img_ids = sorted(os.listdir(dataDir))

    def load_image(self, item):
        img_id = self.img_ids[item]
        imgFile = os.path.join(self.dataDir, img_id, "images", img_id+self.dataSuffix)
        img = cv2.imread(imgFile)
        return img

    def load_annotation(self, item):
        img_id = self.img_ids[item]
        annoFolder = os.path.join(self.dataDir, img_id, "masks")
        masks = load_gt_masks(annoFolder, self.annoSuffix)
        return masks

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


    def __getitem__(self, item):
        img = self.load_image(item)
        height,width,_ = img.shape
        instance_masks = self.load_annotation(item)
        if self.transform is not None:
            img, instance_masks = self.transform(img, instance_masks)

        bboxes_c0, h_c0, w_c0 = load_bboxes(instance_masks, divide_scale = 1., maxi=8.)
        bboxes_c1, h_c1, w_c1 = load_bboxes(instance_masks, divide_scale = 2., maxi=5.)
        bboxes_c2, h_c2, w_c2 = load_bboxes(instance_masks, divide_scale = 4., maxi=5.)
        bboxes_c3, h_c3, w_c3 = load_bboxes(instance_masks, divide_scale = 8., maxi=5.)

        kp0, short0, mid0 = preprocessing.get_ground_truth(bboxes_c0, h_c0, w_c0, num_kps=cfg.NUM_KPS)
        kp1, short1, mid1 = preprocessing.get_ground_truth(bboxes_c1, h_c1, w_c1, num_kps=cfg.NUM_KPS)
        kp2, short2, mid2 = preprocessing.get_ground_truth(bboxes_c2, h_c2, w_c2, num_kps=cfg.NUM_KPS)
        kp3, short3, mid3 = preprocessing.get_ground_truth(bboxes_c3, h_c3, w_c3, num_kps=cfg.NUM_KPS)

        gt_c0 = np.concatenate((kp0, np.transpose(short0, (2,0,1)), np.transpose(mid0, (2,0,1))), 0)
        gt_c1 = np.concatenate((kp1, np.transpose(short1, (2,0,1)), np.transpose(mid1, (2,0,1))), 0)
        gt_c2 = np.concatenate((kp2, np.transpose(short2, (2,0,1)), np.transpose(mid2, (2,0,1))), 0)
        gt_c3 = np.concatenate((kp3, np.transpose(short3, (2,0,1)), np.transpose(mid3, (2,0,1))), 0)

        img[img>255.] = 255.
        img[img<0.] = 0.

        img = torch.FloatTensor(np.transpose(img, (2,0,1))) / 255 - 0.5
        gt_c0 = torch.FloatTensor(gt_c0)
        gt_c1 = torch.FloatTensor(gt_c1)
        gt_c2 = torch.FloatTensor(gt_c2)
        gt_c3 = torch.FloatTensor(gt_c3)

        gt_masks, gt_bboxes = self.load_gt_masks_bboxes(instance_masks)

        return img, gt_c0, gt_c1, gt_c2, gt_c3, gt_masks, gt_bboxes

    def __len__(self):
        return len(self.img_ids)

import torch


def collater(data):
    img = []
    gt_c0 = []
    gt_c1 = []
    gt_c2 = []
    gt_c3 = []
    instance_masks = []
    bboxes_c0 = []
    for sample in data:
        img.append(sample[0])
        gt_c0.append(sample[1])
        gt_c1.append(sample[2])
        gt_c2.append(sample[3])
        gt_c3.append(sample[4])
        instance_masks.append(sample[5])
        bboxes_c0.append(sample[6])
    img = torch.stack(img,0)
    gt_c0 = torch.stack(gt_c0,0)
    gt_c1 = torch.stack(gt_c1,0)
    gt_c2 = torch.stack(gt_c2,0)
    gt_c3 = torch.stack(gt_c3,0)
    return img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0

import torch.nn as nn
import torch
import torch.nn.functional as F
from config import EDGES

class DetectionLossAll(nn.Module):
    def __init__(self, kp_radius):
        super(DetectionLossAll, self).__init__()
        self.kp_radius = kp_radius


    def kp_map_loss(self, pr_kp, gt_kp):
        loss = F.binary_cross_entropy(pr_kp, gt_kp)
        return loss

    def short_offset_loss(self, pr_short, gt_short, gt_kp):
        loss = torch.abs(pr_short - gt_short)/self.kp_radius
        gt_2kps_map = []
        for i in range(gt_kp.shape[1]):
            gt_2kps_map.append(gt_kp[:,i,:,:])
            gt_2kps_map.append(gt_kp[:,i,:,:])

        gt_2kps_map = torch.stack(gt_2kps_map, dim=1)
        loss = loss * gt_2kps_map
        loss = torch.sum(loss)/(torch.sum(gt_2kps_map) + 1e-10)
        return loss

    def mid_offset_loss(self, pr_mid, gt_mid, gt_kp):
        loss = torch.abs(pr_mid - gt_mid)/self.kp_radius
        gt_4edge_map = []
        # bi-direction
        for i, edge in enumerate((EDGES + [edge[::-1] for edge in EDGES])):
            from_kp = edge[0]
            gt_4edge_map.extend([gt_kp[:,from_kp,:,:], gt_kp[:,from_kp,:,:]])
        gt_4edge_map = torch.stack(gt_4edge_map, dim=1)
        loss = loss * gt_4edge_map
        loss = torch.sum(loss)/(torch.sum(gt_4edge_map) + 1e-10)
        return loss

    def forward(self, prediction, groundtruth):
        pr_kp, pr_short, pr_mid = prediction
        gt_kp = groundtruth[:,:5,:,:]
        gt_short = groundtruth[:,5:5+10,:,:]
        gt_mid = groundtruth[:,5+10:,:,:]
        loss_kp = self.kp_map_loss(pr_kp, gt_kp)
        loss_short = self.short_offset_loss(pr_short, gt_short, gt_kp)
        loss_mid = self.mid_offset_loss(pr_mid, gt_mid, gt_kp)
        loss = loss_kp + loss_short + 0.25 * loss_mid
        return loss
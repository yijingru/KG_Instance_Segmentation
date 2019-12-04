from torch.nn.modules import Module
import numpy as np
import config as cfg
import torch.nn.functional as F
import cv2
import torch

class SEG_loss(Module):
    def __init__(self, height, width):
        super(SEG_loss, self).__init__()
        self.height = height
        self.width = width

    def jaccard_numpy(self, a, b):
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        int_ymin = max(a[0],b[0])
        int_xmin = max(a[1],b[1])
        int_ymax = min(a[2],b[2])
        int_xmax = min(a[3],b[3])
        int_h = max(int_ymax-int_ymin, 0.)
        int_w = max(int_xmax-int_xmin, 0.)
        area_inter = int_h*int_w
        union = area_a+area_b-area_inter
        if union <= 2:
            jaccard = 0.
        else:
            jaccard = np.divide(area_inter, union)
        return jaccard

    def forward(self, predictions, gt_masks, gt_boxes):
        # Groundtruth:
        #            boxes:       [batch][obj_boxes]         device(type='cpu')
        #            gt_classes:  [batch][obj_gt_classes]    device(type='cpu')
        #            gt_masks:    [batch][obj_gt_masks]      device(type='cpu')
        # Predictions:
        #            mask_patches: [batch list][List...]    device(type='cuda', index=0)
        #            mask_dets: [batch list][list....]      device(type='cpu')

        mask_patches, mask_dets = predictions
        loss_mask = 0
        run_label = False
        # [~~~~~~~~~~ iterate batch~~~~~~~~~~~~~~~~~~~~~~~]
        for i in range(len(mask_patches)):
            loss_batch = 0
            num_obj = 0

            # [Predict] ~~~~~~~iterate obj~~~~~~~~~~~~~~~~~~~~~
            for j in range(len(mask_patches[i])):
                obj_pr_mask =  mask_patches[i][j]
                obj_p_box =   mask_dets[i][j][:4]

                # [GT] ~~~~~~~~~~~~~~ iterate obj ~~~~~~~~~~~~~~~~~~~
                for i_gt in range(gt_boxes[i].shape[0]):
                    jaccard = self.jaccard_numpy(obj_p_box, torch.Tensor(gt_boxes[i][i_gt]))
                    if jaccard>=0.5:
                        y1,x1,y2,x2 = obj_p_box
                        y1 = np.maximum(0, np.int32(np.round(y1)))
                        x1 = np.maximum(0, np.int32(np.round(x1)))
                        y2 = np.minimum(np.int32(np.round(y2)), self.height - 1)
                        x2 = np.minimum(np.int32(np.round(x2)), self.width - 1)
                        ## Crop the obj_gt_mask from gt_mask
                        obj_gt_mask = gt_masks[i][i_gt][y1:y2,x1:x2]

                        # import cv2
                        #
                        # cv2.imshow('gt', np.uint8(obj_gt_mask * 100))
                        # cv2.imshow('pr', np.uint8(obj_pr_mask.data.cpu().numpy() * 100))
                        # k = cv2.waitKey(0)
                        # if k & 0xFF == ord('q'):
                        #     cv2.destroyAllWindows()
                        #     exit()

                        # h0,w0 = obj_gt_mask.shape
                        h1,w1 = obj_pr_mask.shape

                        obj_gt_mask = cv2.resize(obj_gt_mask, (w1,h1),interpolation=cv2.INTER_NEAREST)

                        obj_gt_mask = torch.Tensor(obj_gt_mask)
                        obj_gt_mask = obj_gt_mask.to(obj_pr_mask.device)


                        h2,w2 = obj_gt_mask.shape
                        assert (h1==h2) and (w1==w2), "[loss.py] mask size does not match!"

                        loss_obj = F.binary_cross_entropy(obj_pr_mask,obj_gt_mask,size_average=True)
                        loss_batch += loss_obj
                        num_obj += 1
                        run_label = True
            if num_obj:
                loss_mask += loss_batch/num_obj

        if run_label:
            loss_mask = loss_mask/len(mask_patches)
        else:
            loss_mask = None
        return loss_mask
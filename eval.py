import torch
import KGnet
import numpy as np
from dataset_kaggle import Kaggle
from dataset_plant import Plant
from dataset_neural import Neural
import argparse
import cv2
import postprocessing
import time
import nms
import eval_parts
import os

def parse_args():
    parser = argparse.ArgumentParser(description="InstanceHeat")
    parser.add_argument("--data_dir", help="data directory", default="../../../../Datasets/root/", type=str)
    parser.add_argument("--resume", help="resume file", default="end_model.pth", type=str)
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms_thresh')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='seg_thresh')
    parser.add_argument('--eval_type', type=str, default='dec', help='seg or dec')
    parser.add_argument("--dataset", help="training dataset", default='neural', type=str)
    args = parser.parse_args()
    return args

class InstanceHeat(object):
    def __init__(self):
        self.model = KGnet.resnet50(pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = {'kaggle': Kaggle, 'plant': Plant, 'neural': Neural}

    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def load_weights(self, resume, dataset):
        self.model.load_state_dict(torch.load(os.path.join('weights_'+dataset, resume)))

    def map_mask_to_image(self, mask, img, color):
        # color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def show_heat_mask(self, mask):
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return heatmap

    def test_inference(self, args, image, bbox_flag=False):
        height, width, c = image.shape

        img_input = cv2.resize(image, (args.input_w, args.input_h))
        img_input = torch.FloatTensor(np.transpose(img_input.copy(), (2, 0, 1))).unsqueeze(0) / 255 - 0.5
        img_input = img_input.to(self.device)

        with torch.no_grad():
            begin = time.time()
            pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
            print("forward time is {:.4f}".format(time.time() - begin))
            pr_kp0, pr_short0, pr_mid0 = pr_c0
            pr_kp1, pr_short1, pr_mid1 = pr_c1
            pr_kp2, pr_short2, pr_mid2 = pr_c2
            pr_kp3, pr_short3, pr_mid3 = pr_c3

        torch.cuda.synchronize()
        skeletons0 = postprocessing.get_skeletons_and_masks(pr_kp0, pr_short0, pr_mid0)
        skeletons1 = postprocessing.get_skeletons_and_masks(pr_kp1, pr_short1, pr_mid1)
        skeletons2 = postprocessing.get_skeletons_and_masks(pr_kp2, pr_short2, pr_mid2)
        skeletons3 = postprocessing.get_skeletons_and_masks(pr_kp3, pr_short3, pr_mid3)

        skeletons0 = postprocessing.refine_skeleton(skeletons0)
        skeletons1 = postprocessing.refine_skeleton(skeletons1)
        skeletons2 = postprocessing.refine_skeleton(skeletons2)
        skeletons3 = postprocessing.refine_skeleton(skeletons3)

        bboxes = postprocessing.gather_skeleton(skeletons0, skeletons1, skeletons2, skeletons3)
        bboxes = nms.non_maximum_suppression_numpy(bboxes, nms_thresh=args.nms_thresh)
        if bbox_flag:
            return bboxes
        if bboxes is None:
            return None

        with torch.no_grad():
            predictions = self.model.forward_seg(feat_seg, [bboxes])
        torch.cuda.synchronize()
        predictions = self.post_processing(args, predictions, width, height)
        return predictions

    def post_processing(self, args, predictions, image_w, image_h):
        if predictions is None:
            return predictions
        out_masks = []
        out_dets = []
        mask_patches, mask_dets = predictions
        for mask_b_patches, mask_b_dets in zip(mask_patches, mask_dets):
            for mask_n_patch, mask_n_det in zip(mask_b_patches, mask_b_dets):
                mask_patch = mask_n_patch.data.cpu().numpy()
                mask_det = mask_n_det.data.cpu().numpy()
                y1, x1, y2, x2, conf = mask_det
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), args.input_h - 1)
                x2 = np.minimum(np.int32(np.round(x2)), args.input_w - 1)

                mask = np.zeros((args.input_h, args.input_w), dtype=np.float32)
                mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))

                mask[y1:y2, x1:x2] = mask_patch
                mask = cv2.resize(mask, (image_w, image_h))
                mask = np.where(mask >= args.seg_thresh, 1, 0)

                y1 = float(y1) / args.input_h * image_h
                x1 = float(x1) / args.input_w * image_w
                y2 = float(y2) / args.input_h * image_h
                x2 = float(x2) / args.input_w * image_w

                out_masks.append(mask)
                out_dets.append([y1,x1,y2,x2, conf])
        return [np.asarray(out_masks, np.float32), np.asarray(out_dets, np.float32)]


    def instance_segmentation_evaluation(self, args, ov_thresh=0.5, use_07_metric=False):
        self.load_weights(resume=args.resume, dataset=args.dataset)
        self.model.eval()
        self.model = self.model.to(self.device)

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir, phase='test')

        all_tp = []
        all_fp = []
        all_scores = []
        temp_overlaps = []
        npos = 0
        for index in range(len(dsets)):
            print('processing {}/{} images'.format(index, len(dsets)))
            img = dsets.load_image(index)
            predictions = self.test_inference(args, img)
            if predictions is None:
                npos += len(dsets.load_annotation(index, type='bbox'))
                continue
            pr_masks, pr_dets = predictions

            fp, tp, all_scores, npos, temp_overlaps = eval_parts.seg_evaluation(index=index,
                                                                            dsets=dsets,
                                                                            BB_masks=pr_masks,
                                                                            BB_dets=pr_dets,
                                                                            all_scores=all_scores,
                                                                            npos=npos,
                                                                            temp_overlaps=temp_overlaps,
                                                                            ov_thresh=ov_thresh)

            all_fp.extend(fp)
            all_tp.extend(tp)
        # step5: compute precision recall
        all_fp = np.asarray(all_fp)
        all_tp = np.asarray(all_tp)
        all_scores = np.asarray(all_scores)
        sorted_ind = np.argsort(-all_scores)
        all_fp = all_fp[sorted_ind]
        all_tp = all_tp[sorted_ind]
        all_fp = np.cumsum(all_fp)
        all_tp = np.cumsum(all_tp)
        rec = all_tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)
        ap = eval_parts.voc_ap(rec, prec, use_07_metric=use_07_metric)
        print("ap@{} is {}".format(ov_thresh, ap))
        print("temp overlaps = {}".format(np.mean(temp_overlaps)))
        return ap, np.mean(temp_overlaps)

    def detection_evaluation(self, args, ov_thresh=0.5, use_07_metric=False):
        self.load_weights(resume=args.resume, dataset=args.dataset)
        self.model.eval()
        self.model = self.model.to(self.device)

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir, phase='test')

        all_tp = []
        all_fp = []
        all_scores = []
        npos = 0
        for index in range(len(dsets)):
            print('processing {}/{} images'.format(index, len(dsets)))
            img = dsets.load_image(index)
            height, width, c = img.shape

            bboxes = self.test_inference(args, img, bbox_flag = True)

            if bboxes is None:
                npos += len(dsets.load_annotation(index, type='bbox'))
                continue

            bboxes = np.asarray(bboxes, np.float32)

            bboxes[:, 0] = bboxes[:, 0] / args.input_h * height
            bboxes[:, 1] = bboxes[:, 1] / args.input_w * width
            bboxes[:, 2] = bboxes[:, 2] / args.input_h * height
            bboxes[:, 3] = bboxes[:, 3] / args.input_w * width

            fp, tp, all_scores, npos = eval_parts.bbox_evaluation(index=index,
                                                                  dsets=dsets,
                                                                  BB_bboxes=bboxes,
                                                                  all_scores=all_scores,
                                                                  npos=npos,
                                                                  ov_thresh=ov_thresh)
            all_fp.extend(fp)
            all_tp.extend(tp)
        # step5: compute precision recall
        all_fp = np.asarray(all_fp)
        all_tp = np.asarray(all_tp)
        all_scores = np.asarray(all_scores)
        sorted_ind = np.argsort(-all_scores)
        all_fp = all_fp[sorted_ind]
        all_tp = all_tp[sorted_ind]
        all_fp = np.cumsum(all_fp)
        all_tp = np.cumsum(all_tp)
        rec = all_tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)
        ap = eval_parts.voc_ap(rec, prec, use_07_metric=use_07_metric)
        print("ap@{} is {}".format(ov_thresh, ap))
        return ap



def run_seg_ap(object_is, args):
    print('evaluating segmentation using PASCAL2010 metric')
    thresh = np.linspace(0.5, 0.95, 10)
    ap_list = []
    iou_list = []
    for v in thresh:
        ap, iou = object_is.instance_segmentation_evaluation(args, ov_thresh=v, use_07_metric=False)
        ap_list.append(ap*100)
        iou_list.append(iou*100)
    np.savetxt(os.path.join('weights_'+args.dataset, 'seg_ap_list.txt'), ap_list, fmt='%.4f')
    np.savetxt(os.path.join('weights_'+args.dataset, 'seg_iou_list.txt'), iou_list, fmt='%.4f')



def run_dec_ap(object_is, args):
    print('evaluating detection using PASCAL2010 metric')
    thresh = np.linspace(0.5, 0.95, 10)
    ap_list = []
    for v in thresh:
        ap = object_is.detection_evaluation(args, ov_thresh=v, use_07_metric=False)
        ap_list.append(ap*100)
    np.savetxt(os.path.join('weights_'+args.dataset, 'dec_ap_list.txt'), ap_list, fmt='%.4f')

if __name__ == '__main__':
    args = parse_args()
    object_is = InstanceHeat()
    if args.eval_type == 'seg':
        run_seg_ap(object_is, args)
    else:
        run_dec_ap(object_is, args)

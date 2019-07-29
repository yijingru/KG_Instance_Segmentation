import torch
import os
import net
import numpy as np
import transforms
from dataset import Kaggle
import argparse
import cv2
from loss import DetectionLossAll
import postprocessing
import config as cfg
import time
import show_ground_truth
import nms
import evaluation
import seg_loss
from collater import collater

def parse_args():
    parser = argparse.ArgumentParser(description="InstanceHeat")
    parser.add_argument("--data_dir", help="data directory",
                        default="../Datasets/kaggle/",
                        type=str)
    parser.add_argument("--resume", help="resume file", default="end_model.pth", type=str)
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument("--workers", help="workers number", default=4, type=int)
    parser.add_argument("--batch_size", help="batch size", default=2, type=int)
    parser.add_argument("--epochs", help="epochs", default=100, type=int)
    parser.add_argument("--start_epoch", help="start_epoch", default=0, type=int)
    parser.add_argument("--lr", help="learning_rate", default=0.0001, type=int)
    parser.add_argument("--data_parallel", help="data parallel", default=False, type=bool)
    args = parser.parse_args()
    return args



class InstanceHeat(object):
    def __init__(self):
        self.model = net.resnet50(pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def load_weights(self, resume):
        self.model.load_state_dict(torch.load(resume))

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

    def train(self, args):
        if not os.path.exists("weights"):
            os.mkdir("weights")

        self.model = self.model.to(self.device)

        self.model.train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)

        loss_dec = DetectionLossAll(kp_radius=cfg.KP_RADIUS)
        loss_seg = seg_loss.SEG_loss(height=args.image_height, width=args.input_w)

        data_trans = {'train': transforms.Compose([transforms.ConvertImgFloat(),
                                                   transforms.PhotometricDistort(),
                                                   transforms.Expand(max_scale=2, mean=(0, 0, 0)),
                                                   transforms.RandomMirror_w(),
                                                   transforms.RandomMirror_h(),
                                                   transforms.Resize(h=args.input_h, w=args.input_w)]),

                      'val': transforms.Compose([transforms.ConvertImgFloat(),
                                                 transforms.Resize(h=args.input_h, w=args.input_w)])}

        dsets = {x: Kaggle(data_dir=args.data_dir,
                                   phase=x,
                                   transform=data_trans[x])
                 for x in ['train', 'val']}


        # for i in range(100):
        #     show_ground_truth.show_input(dsets.__getitem__(i))


        train_loader = torch.utils.data.DataLoader(dsets['train'],
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   collate_fn = collater)


        val_loader = torch.utils.data.DataLoader(dsets['val'],
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 shuffle=False,
                                                 collate_fn = collater)


        train_loss_dict = []
        val_loss_dict = []
        for epoch in range(args.start_epoch, args.epochs):
            print('Epoch {}/{}'.format(epoch, args.epochs - 1))
            print('-' * 10)
            # scheduler.step()

            train_epoch_loss = self.training(train_loader,loss_dec,loss_seg,optimizer,epoch, dsets['train'])
            train_loss_dict.append(train_epoch_loss)

            val_epoch_loss = self.validating(val_loader,loss_dec,loss_seg, epoch, dsets['val'])
            val_loss_dict.append(val_epoch_loss)

            np.savetxt('train_loss.txt', train_loss_dict, fmt='%.6f')
            np.savetxt('val_loss.txt', val_loss_dict, fmt='%.6f')

            if epoch % 5 == 0 and epoch >0:
                torch.save(self.model.state_dict(), os.path.join('weights', '{:d}_{:.4f}_model.pth'.format(epoch, train_epoch_loss)))
            torch.save(self.model.state_dict(), os.path.join('weights', args.resume))

    def training(self, train_loader, loss_dec, loss_seg, optimizer, epoch, dsets):
        self.model.train()
        running_loss = 0.0

        for data in train_loader:
            img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0  = data
            img = img.to(self.device)
            gt_c0 = gt_c0.to(self.device)
            gt_c1 = gt_c1.to(self.device)
            gt_c2 = gt_c2.to(self.device)
            gt_c3 = gt_c3.to(self.device)

            optimizer.zero_grad()

            with torch.enable_grad():
                pr_c0, pr_c1, pr_c2, pr_c3, predictions = self.model(img, bboxes_c0)
                loss1 = loss_dec(pr_c0, gt_c0)+loss_dec(pr_c1, gt_c1)+loss_dec(pr_c2, gt_c2)+loss_dec(pr_c3, gt_c3)
                loss2 = loss_seg(predictions, instance_masks, bboxes_c0)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(dsets)
        print('{} Loss: {:.6}'.format(epoch, epoch_loss))
        return epoch_loss

    def validating(self, val_loader, loss_dec, loss_seg, epoch, dsets):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0  = data
                img = img.to(self.device)
                gt_c0 = gt_c0.to(self.device)
                gt_c1 = gt_c1.to(self.device)
                gt_c2 = gt_c2.to(self.device)
                gt_c3 = gt_c3.to(self.device)
                pr_c0, pr_c1, pr_c2, pr_c3, predictions = self.model(img, bboxes_c0)
                loss1 = loss_dec(pr_c0, gt_c0)+loss_dec(pr_c1, gt_c1)+loss_dec(pr_c2, gt_c2)+loss_dec(pr_c3, gt_c3)
                loss2 = loss_seg(predictions, instance_masks, bboxes_c0)
                loss = loss1 + loss2
                running_loss += loss.item()
        epoch_loss = running_loss / len(dsets)
        print('Valid {} Loss: {:.6}'.format(epoch, epoch_loss))
        return epoch_loss


    def imshow_kp(self, kp, img_in):
        h,w = kp.shape[2:]
        img = cv2.resize(img_in, (w, h))
        colors = [(0,0,0.9),(0.9,0,0),(0.9,0,0.9),(0.9,0.9,0), (0.2,0.9,0.9)]
        for i in range(kp.shape[1]):
            img = self.map_mask_to_image(kp[0,i,:,:], img, color=colors[i])
        return img

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
        bboxes = nms.non_maximum_suppression_numpy(bboxes, nms_thresh=0.5)
        if bbox_flag:
            return bboxes
        if bboxes is None:
            return None

        with torch.no_grad():
            predictions = self.model.forward_seg(feat_seg, [bboxes])
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
                mask = np.where(mask >= 0.5, 1, 0)

                y1 = float(y1) / args.input_h * image_h
                x1 = float(x1) / args.input_w * image_w
                y2 = float(y2) / args.input_h * image_h
                x2 = float(x2) / args.input_w * image_w

                out_masks.append(mask)
                out_dets.append([y1,x1,y2,x2, conf])
        return [np.asarray(out_masks, np.float32), np.asarray(out_dets, np.float32)]



    def imshow_instance_segmentation(self, masks, dets, out_img, img_id=None, save_flag=False, show_box=False):
        for mask, det in zip(masks, dets):
            color = np.random.rand(3)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mskd = out_img * mask
            if show_box:
                y1,x1,y2,x2,conf = det
                cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 1, 1)
                cv2.putText(out_img, "{:.4f}".format(conf), (int(x1),int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, 1)
            clmsk = np.ones(mask.shape) * mask
            clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
            clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
            clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
            out_img = out_img + 1 * clmsk - 1 * mskd
        if save_flag:
            cv2.imwrite(os.path.join("save_result", img_id+".png"), np.uint8(out_img))
        cv2.imshow('out_img', np.uint8(out_img))
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(1)



    def test(self, args, save_flag=False):
        self.load_weights(resume=args.resume)
        self.model = self.model.to(self.device)
        self.model.eval()

        if not os.path.exists("save_result") and save_flag is True:
            os.mkdir("save_result")

        dsets = Kaggle(data_dir=args.data_dir, phase='test')

        for index in range(len(dsets)):
            img = dsets.load_image(index)
            predictions = self.test_inference(args, img)
            if predictions is None:
                continue
            mask_patches, mask_dets = predictions
            self.imshow_instance_segmentation(mask_patches, mask_dets, out_img=img.copy(),
                                              img_id=dsets.img_ids[index], save_flag= False)



    def detection_evaluation(self, args, ov_thresh=0.5, use_07_metric=True):
        self.load_weights(resume=args.resume)
        self.model.eval()
        self.model = self.model.to(self.device)

        dsets = Kaggle(data_dir=args.data_dir, phase='test')

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

            fp, tp, all_scores, npos = evaluation.bbox_evaluation(index=index,
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
        ap = evaluation.voc_ap(rec, prec, use_07_metric=use_07_metric)
        print("ap@{} is {}".format(ov_thresh, ap))


    def instance_segmentation_evaluation(self, args, ov_thresh=0.5, use_07_metric=True):
        self.load_weights(resume=args.resume)
        self.model.eval()
        self.model = self.model.to(self.device)

        dsets = Kaggle(data_dir=args.data_dir, phase='test')

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

            fp, tp, all_scores, npos, temp_overlaps = evaluation.seg_evaluation(index=index,
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
        ap = evaluation.voc_ap(rec, prec, use_07_metric=use_07_metric)
        print("ap@{} is {}".format(ov_thresh, ap))
        print("temp overlaps = {}".format(np.mean(temp_overlaps)))

if __name__ == '__main__':
    args = parse_args()
    object_is = InstanceHeat()
    object_is.train(args)
    # object_is.test(args)
    # object_is.detection_evaluation(args,ov_thresh=0.5)
    # object_is.detection_evaluation(args,ov_thresh=0.7)
    # object_is.instance_segmentation_evaluation(args, ov_thresh=0.5)
    # object_is.instance_segmentation_evaluation(args, ov_thresh=0.7)

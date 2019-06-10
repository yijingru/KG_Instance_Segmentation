import torch
import os
import net
import numpy as np
import transforms
from dataset import NeucleiDataset, load_gt_bboxes, load_gt_masks, mask_iou
import argparse
import cv2
import glob
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
    parser.add_argument("--trainDir", help="data directory",
                        default="/home/grace/PycharmProjects/DataSets/kaggle/train",
                        type=str)
    parser.add_argument("--testDir", help="test directory",
                        default="/home/grace/PycharmProjects/DataSets/kaggle/test",
                        type=str)
    parser.add_argument("--valDir", help="test directory",
                        default="/home/grace/PycharmProjects/DataSets/kaggle/val",
                        type=str)
    parser.add_argument("--resume", help="resume file",
                        default="end_model.pth",
                        type=str)
    parser.add_argument("--image_height", help="image_height", default=512, type=int)
    parser.add_argument("--image_width", help="image_width", default=512, type=int)
    parser.add_argument("--dataSuffix", help="data suffix", default=".png", type=str)
    parser.add_argument("--annoSuffix", help="anno suffix", default=".png", type=str)
    parser.add_argument("--workers", help="workers number", default=4, type=int)
    parser.add_argument("--batch_size", help="batch size", default=2, type=int)
    parser.add_argument("--epochs", help="epochs", default=100, type=int)
    parser.add_argument("--start_epoch", help="start_epoch", default=0, type=int)
    parser.add_argument("--lr", help="learning_rate", default=0.0001, type=int)
    parser.add_argument("--data_parallel", help="data parrallel", default=False, type=bool)
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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)

        loss_dec = DetectionLossAll(kp_radius=cfg.KP_RADIUS)
        loss_seg = seg_loss.SEG_loss(height=args.image_height, width=args.image_width)

        train_data_transform = transforms.Compose([transforms.ConvertImgFloat(),
                                                   transforms.PhotometricDistort(),
                                                   transforms.Expand(max_scale=2, mean=(0, 0, 0)),
                                                   transforms.RandomMirror_w(),
                                                   transforms.RandomMirror_h(),
                                                   transforms.Resize(h=args.image_height, w=args.image_width)])

        val_data_transform = transforms.Compose([transforms.ConvertImgFloat(),
                                                 transforms.Resize(h=args.image_height, w=args.image_width)])

        train_dsets = NeucleiDataset(dataDir=args.trainDir, annoDir=None,
                                     dataSuffix=args.dataSuffix, annoSuffix=args.annoSuffix,
                                     transform=train_data_transform)

        val_dsets = NeucleiDataset(dataDir=args.valDir, annoDir=None,
                                   dataSuffix=args.dataSuffix, annoSuffix=args.annoSuffix,
                                   transform=val_data_transform)

        # for i in range(100):
        #     show_ground_truth.show_input(dsets.__getitem__(i))


        train_loader = torch.utils.data.DataLoader(train_dsets,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   collate_fn = collater)


        val_loader = torch.utils.data.DataLoader(val_dsets,
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
            scheduler.step()

            train_epoch_loss = self.training(train_loader,loss_dec,loss_seg,optimizer,epoch, train_dsets)
            train_loss_dict.append(train_epoch_loss)

            val_epoch_loss = self.validating(val_loader,loss_dec,loss_seg, epoch, val_dsets)
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

    def test(self, args):
        self.load_weights(resume=args.resume)
        self.model.eval()
        self.model = self.model.to(self.device)

        fileLists = os.listdir(args.testDir)
        for img_id in sorted(fileLists):
            imgDir = os.path.join(args.testDir, img_id, "images", img_id+args.dataSuffix)
            img = cv2.imread(imgDir)

            h,w,c = img.shape

            img_input = cv2.resize(img, (args.image_width, args.image_height))
            img_input =  torch.FloatTensor(np.transpose(img_input.copy(), (2,0,1))).unsqueeze(0)/255 - 0.5
            img_input = img_input.to(self.device)

            with torch.no_grad():
                begin = time.time()
                pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
                print('running time is {:.4f}'.format(time.time()-begin))
                pr_kp0, pr_short0, pr_mid0 = pr_c0
                pr_kp1, pr_short1, pr_mid1 = pr_c1
                pr_kp2, pr_short2, pr_mid2 = pr_c2
                pr_kp3, pr_short3, pr_mid3 = pr_c3

            pr_kp0 = pr_kp0.data.cpu().numpy()
            pr_kp1 = pr_kp1.data.cpu().numpy()
            pr_kp2 = pr_kp2.data.cpu().numpy()
            pr_kp3 = pr_kp3.data.cpu().numpy()
            pr_short0 = np.transpose(pr_short0.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_short1 = np.transpose(pr_short1.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_short2 = np.transpose(pr_short2.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_short3 = np.transpose(pr_short3.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_mid0 = np.transpose(pr_mid0.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_mid1 = np.transpose(pr_mid1.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_mid2 = np.transpose(pr_mid2.data[0,:,:,:].cpu().numpy(), (1,2,0))
            pr_mid3 = np.transpose(pr_mid3.data[0,:,:,:].cpu().numpy(), (1,2,0))

            img_c0 = self.imshow_kp(pr_kp0, img)
            img_c1 = self.imshow_kp(pr_kp1, img)
            img_c2 = self.imshow_kp(pr_kp2, img)
            img_c3 = self.imshow_kp(pr_kp3, img)

            # for i_kp in range(5):
            #     offset = pr_short3[:,:,i_kp*2:i_kp*2+2]
            #     map = np.sqrt(offset[:,:,0]**2+offset[:,:,1]**2)
            #     cv2.imshow('short', np.uint8((map*200)))
            #
            # for i_edge in range(20):
            #     offset = pr_mid3[:,:,i_edge*2:i_edge*2+2]
            #     map = np.sqrt(offset[:,:,0]**2+offset[:,:,1]**2)
            #     cv2.imshow('mid', np.uint8((map*200)))

            cv2.imshow('img_c0', img_c0)
            cv2.imshow('img_c1', img_c1)
            cv2.imshow('img_c2', img_c2)
            cv2.imshow('img_c3', img_c3)

            cv2.imwrite('img_c0.png', img_c0)
            cv2.imwrite('img_c1.png', img_c1)
            cv2.imwrite('img_c2.png', img_c2)
            cv2.imwrite('img_c3.png', img_c3)

            k=cv2.waitKey(0)
            if k&0xFF==ord('q'):
                cv2.destroyAllWindows()
                exit()

        print('wait')


    def imshow_skeleton(self, skeletons, img, color):
        for skeleton in skeletons:
            out = []
            for point in skeleton:
                x,y,conf = point
                if x==0 and y==0:
                    continue
                out.append([int(x),int(y),conf])

            for i in range(len(out)):
                for j in range(len(out)):
                    if i!= j:
                        x1,y1,conf1 = out[i]
                        x2,y2,conf2 = out[j]
                        cv2.line(img,(x2,y2),(x1,y1),color=(255,255,255), thickness=1, lineType=1)
        return img

    def test1(self, args, conf_thresh=0.01, seg_thresh=0.5, nms_thresh=0.5):
        self.load_weights(resume=args.resume)
        self.model = self.model.to(self.device)
        self.model.eval()

        if not os.path.exists("save_results"):
            os.mkdir("save_results")

        fileLists = os.listdir(args.testDir)
        for img_id in sorted(fileLists):
            imgDir = os.path.join(args.testDir, img_id, "images", img_id+args.dataSuffix)
            img = cv2.imread(imgDir)
            h,w,c = img.shape

            img = cv2.resize(img, (args.image_width, args.image_height))

            img_input =  torch.FloatTensor(np.transpose(img.copy(), (2,0,1))).unsqueeze(0)/255 - 0.5
            img_input = img_input.to(self.device)
            with torch.no_grad():
                begin = time.time()
                pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
                print("forward time is {:.4f}".format(time.time()-begin))
                pr_kp0, pr_short0, pr_mid0 = pr_c0
                pr_kp1, pr_short1, pr_mid1 = pr_c1
                pr_kp2, pr_short2, pr_mid2 = pr_c2
                pr_kp3, pr_short3, pr_mid3 = pr_c3

            skeletons0 = postprocessing.get_skeletons_and_masks(pr_kp0, pr_short0, pr_mid0)
            skeletons1 = postprocessing.get_skeletons_and_masks(pr_kp1, pr_short1, pr_mid1)
            skeletons2 = postprocessing.get_skeletons_and_masks(pr_kp2, pr_short2, pr_mid2)
            skeletons3 = postprocessing.get_skeletons_and_masks(pr_kp3, pr_short3, pr_mid3)

            skeletons0 = postprocessing.refine_skeleton(skeletons0)
            skeletons1 = postprocessing.refine_skeleton(skeletons1)
            skeletons2 = postprocessing.refine_skeleton(skeletons2)
            skeletons3 = postprocessing.refine_skeleton(skeletons3)

            img_c0 = cv2.resize(img, (pr_kp0.shape[3],pr_kp0.shape[2]))
            img_c1 = cv2.resize(img, (pr_kp1.shape[3],pr_kp1.shape[2]))
            img_c2 = cv2.resize(img, (pr_kp2.shape[3],pr_kp2.shape[2]))
            img_c3 = cv2.resize(img, (pr_kp3.shape[3],pr_kp3.shape[2]))


            cv2.imwrite('img_c0.png', img_c0)
            cv2.imwrite('img_c1.png', img_c1)
            cv2.imwrite('img_c2.png', img_c2)
            cv2.imwrite('img_c3.png', img_c3)

            color = (200, 255, 255)
            img_c0 = self.imshow_skeleton(skeletons0, img_c0, color)
            img_c1 = self.imshow_skeleton(skeletons1, img_c1, color)
            img_c2 = self.imshow_skeleton(skeletons2, img_c2, color)
            img_c3 = self.imshow_skeleton(skeletons3, img_c3, color)




            pr_kp0 = pr_kp0.data.cpu().numpy()
            pr_kp1 = pr_kp1.data.cpu().numpy()
            pr_kp2 = pr_kp2.data.cpu().numpy()
            pr_kp3 = pr_kp3.data.cpu().numpy()

            img_c0 = self.imshow_kp(pr_kp0, img_c0)
            img_c1 = self.imshow_kp(pr_kp1, img_c1)
            img_c2 = self.imshow_kp(pr_kp2, img_c2)
            img_c3 = self.imshow_kp(pr_kp3, img_c3)

            cv2.imshow('img_c0', img_c0)
            cv2.imshow('img_c1', img_c1)
            cv2.imshow('img_c2', img_c2)
            cv2.imshow('img_c3', img_c3)

            cv2.imwrite('skeleton_img_c0.png', img_c0)
            cv2.imwrite('skeleton_img_c1.png', img_c1)
            cv2.imwrite('skeleton_img_c2.png', img_c2)
            cv2.imwrite('skeleton_img_c3.png', img_c3)

            k = cv2.waitKey(0)&0xFF
            if k==ord('q'):
                cv2.destroyAllWindows()
                exit()




    def test5(self, args):
        self.load_weights(resume=args.resume)
        self.model = self.model.to(self.device)
        self.model.eval()

        if not os.path.exists("save_result"):
            os.mkdir("save_result")

        fileLists = os.listdir(args.testDir)
        for img_id in sorted(fileLists):
            imgDir = os.path.join(args.testDir, img_id, "images", img_id+args.dataSuffix)
            img = cv2.imread(imgDir)
            out_img = img.copy()
            height, width, _ = out_img.shape
            img = cv2.resize(img, (args.image_width, args.image_height))
            img_input =  torch.FloatTensor(np.transpose(img.copy(), (2,0,1))).unsqueeze(0)/255 - 0.5
            img_input = img_input.to(self.device)


            with torch.no_grad():
                begin = time.time()
                pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
                print("forward time is {:.4f}".format(time.time()-begin))
                pr_kp0, pr_short0, pr_mid0 = pr_c0
                pr_kp1, pr_short1, pr_mid1 = pr_c1
                pr_kp2, pr_short2, pr_mid2 = pr_c2
                pr_kp3, pr_short3, pr_mid3 = pr_c3

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

            if bboxes is None:
                continue

            with torch.no_grad():
                predictions = self.model.forward_seg(feat_seg, [bboxes])

            (mask_patches, mask_dets) = predictions
            for i_batch in range(len(mask_patches)):
                for i_obj in range(len(mask_patches[i_batch])):
                    mask_patch = mask_patches[i_batch][i_obj].data.cpu().numpy()
                    mask_det = mask_dets[i_batch][i_obj].data.cpu().numpy()
                    [y1, x1, y2, x2, conf] = mask_det
                    y1 = np.maximum(0, np.int32(np.round(y1)))
                    x1 = np.maximum(0, np.int32(np.round(x1)))
                    y2 = np.minimum(np.int32(np.round(y2)), args.image_height - 1)
                    x2 = np.minimum(np.int32(np.round(x2)), args.image_width - 1)

                    mask = np.zeros((args.image_height, args.image_width), dtype=np.float32)
                    mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))

                    mask[y1:y2, x1:x2] = mask_patch
                    mask = cv2.resize(mask, (width, height))
                    mask = np.where(mask >= 0.5, 1, 0)

                    y1 = int(float(y1)/args.image_height*height)
                    x1 = int(float(x1)/args.image_width*width)
                    y2 = int(float(y2)/args.image_height*height)
                    x2 = int(float(x2)/args.image_width*width)


                    color = np.random.rand(3)
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    mskd = out_img * mask

                    clmsk = np.ones(mask.shape) * mask
                    clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                    clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                    clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                    out_img = out_img + 1 * clmsk - 1 * mskd
                    # cv2.rectangle(out_img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=[0, 255, 0], thickness=1,
                    #               lineType=1)
                    # cv2.putText(out_img, "{:.4f}".format(conf), (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,0,255), thickness=1, lineType=1)
                cv2.imwrite(os.path.join("save_result",img_id+".png"), np.uint8(out_img))
                # cv2.imshow('img', np.uint8(img))
                # cv2.imshow('out_img', np.uint8(out_img))
                # k = cv2.waitKey(0)
                # if k & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                #     exit(1)


    def detection_evaluation(self, args, ov_thresh=0.5, use_07_metric=True):
        self.load_weights(resume=args.resume)
        self.model.eval()
        self.model = self.model.to(self.device)

        all_tp = []
        all_fp = []
        all_scores = []
        npos = 0
        fileLists = os.listdir(args.testDir)
        for img_id in sorted(fileLists):
            imgDir = os.path.join(args.testDir, img_id, "images", img_id+args.dataSuffix)
            img = cv2.imread(imgDir)

            height, width, c = img.shape

            img_input = cv2.resize(img, (args.image_width, args.image_height))
            img_input =  torch.FloatTensor(np.transpose(img_input.copy(), (2,0,1))).unsqueeze(0)/255 - 0.5
            img_input = img_input.to(self.device)

            with torch.no_grad():
                begin = time.time()
                pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
                print("forward time is {:.4f}".format(time.time()-begin))
                pr_kp0, pr_short0, pr_mid0 = pr_c0
                pr_kp1, pr_short1, pr_mid1 = pr_c1
                pr_kp2, pr_short2, pr_mid2 = pr_c2
                pr_kp3, pr_short3, pr_mid3 = pr_c3

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

            if bboxes is None:
                continue

            bboxes = np.asarray(bboxes, np.float32)

            bboxes[:,0]= bboxes[:,0]/args.image_height*height
            bboxes[:,1]= bboxes[:,1]/args.image_width*width
            bboxes[:,2]= bboxes[:,2]/args.image_height*height
            bboxes[:,3]= bboxes[:,3]/args.image_width*width


            pr_conf = bboxes[:, 4]
            pr_bboxes = bboxes[:, :4]
            sorted_ind = np.argsort(-pr_conf)
            pr_bboxes = pr_bboxes[sorted_ind, :]
            pr_conf = pr_conf[sorted_ind]
            all_scores.extend(pr_conf)

            # Step2: initialization of evaluations
            nd = pr_bboxes.shape[0]
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            BBGT_box = load_gt_bboxes(os.path.join(args.testDir, img_id, "masks"), args.annoSuffix)
            nd_gt = BBGT_box.shape[0]
            det_flag = [False] * nd_gt
            npos = npos + nd_gt

            for d in range(nd):
                bb = pr_bboxes[d, :].astype(float)
                ovmax = -np.inf
                BBGT = BBGT_box.astype(float)
                jmax = -1
                if BBGT.shape[0]>0:
                    iymin = np.maximum(BBGT[:, 0], bb[0])
                    ixmin = np.maximum(BBGT[:, 1], bb[1])
                    iymax = np.minimum(BBGT[:, 2], bb[2])
                    ixmax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    union = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                             (BBGT[:, 2] - BBGT[:, 0]) *
                             (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / union
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ov_thresh:
                    if not det_flag[jmax]:
                        tp[d] = 1.
                        det_flag[jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.
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

        all_tp = []
        all_fp = []
        all_scores = []
        temp_overlaps = []
        npos = 0

        fileLists = os.listdir(args.testDir)
        for img_id in sorted(fileLists):
            imgDir = os.path.join(args.testDir, img_id, "images", img_id+args.dataSuffix)
            img = cv2.imread(imgDir)
            height,width,_ = img.shape
            img_input = cv2.resize(img, (args.image_width, args.image_height))
            img_input =  torch.FloatTensor(np.transpose(img_input.copy(), (2,0,1))).unsqueeze(0)/255 - 0.5
            img_input = img_input.to(self.device)

            with torch.no_grad():
                # begin = time.time()
                pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
                #print("forward time is {:.4f}".format(time.time()-begin))
                pr_kp0, pr_short0, pr_mid0 = pr_c0
                pr_kp1, pr_short1, pr_mid1 = pr_c1
                pr_kp2, pr_short2, pr_mid2 = pr_c2
                pr_kp3, pr_short3, pr_mid3 = pr_c3

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

            if bboxes is None:
                img_id = imgDir.split('/')[-1].split('.')[0]
                BBGT_mask = load_gt_masks(os.path.join(args.testDir, img_id, "masks"), args.annoSuffix)
                nd_gt = BBGT_mask.shape[0]
                npos = npos + nd_gt
                continue


            with torch.no_grad():
                predictions = self.model.forward_seg(feat_seg, [bboxes])
            (mask_patches, mask_dets) = predictions

            # for batches
            for b_mask_patches, b_mask_dets in zip(mask_patches, mask_dets):
                nd = len(b_mask_dets)
                BB_conf = []
                BB_mask = []
                for d in range(nd):
                    d_mask_det = b_mask_dets[d].data.cpu().numpy()
                    d_mask_patch = b_mask_patches[d].data.cpu().numpy()
                    [y1, x1, y2, x2, conf] = d_mask_det
                    y1 = np.maximum(0, np.int32(np.round(y1)))
                    x1 = np.maximum(0, np.int32(np.round(x1)))
                    y2 = np.minimum(np.int32(np.round(y2)), args.image_height - 1)
                    x2 = np.minimum(np.int32(np.round(x2)), args.image_width - 1)
                    mask = np.zeros((args.image_height, args.image_width), dtype=np.float32)
                    mask_patch = cv2.resize(d_mask_patch, (x2 - x1, y2 - y1))
                    mask[y1:y2, x1:x2] = mask_patch
                    mask = cv2.resize(mask, (width, height))
                    mask = np.where(mask >= 0.5, 1, 0)

                    BB_conf.append(conf)
                    BB_mask.append(mask)

                BB_conf = np.asarray(BB_conf, dtype=np.float32)
                BB_mask = np.asarray(BB_mask, dtype=np.float32)
                # Step2: sort detections according to the confidences
                sorted_ind = np.argsort(-BB_conf)
                BB_mask = BB_mask[sorted_ind, :, :]
                BB_conf = BB_conf[sorted_ind]
                all_scores.extend(BB_conf)

                # Step2: intialzation of evaluations
                nd = BB_mask.shape[0]
                tp = np.zeros(nd)
                fp = np.zeros(nd)

                img_id = imgDir.split('/')[-1].split('.')[0]
                BBGT_mask = load_gt_masks(os.path.join(args.testDir, img_id, "masks"), args.annoSuffix)
                nd_gt = BBGT_mask.shape[0]
                det_flag = [False] * nd_gt
                npos = npos + nd_gt

                for d in range(nd):
                    d_BB_mask = BB_mask[d, :, :]
                    ovmax = -np.inf
                    jmax = -1
                    for ind2 in range(len(BBGT_mask)):
                        gt_mask = BBGT_mask[ind2]
                        overlaps = mask_iou(d_BB_mask, gt_mask)
                        if overlaps > ovmax:
                            ovmax = overlaps
                            jmax = ind2

                    if ovmax > ov_thresh:
                        if not det_flag[jmax]:
                            tp[d] = 1.
                            det_flag[jmax] = 1
                            temp_overlaps.append(ovmax)
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.
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
    #object_is.train(args)
    object_is.test(args)
    # object_is.test1(args)
    # object_is.test5(args)
    # object_is.detection_evaluation(args,ov_thresh=0.5)
    # object_is.detection_evaluation(args,ov_thresh=0.7)
    # object_is.instance_segmentation_evaluation(args, ov_thresh=0.5)
    # object_is.instance_segmentation_evaluation(args, ov_thresh=0.7)

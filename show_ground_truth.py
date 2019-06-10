
import numpy as np
import cv2

def split_gt(gt):
    gt = gt.data.cpu().numpy()
    kp = gt[:5,:,:]
    short = np.transpose(gt[5:5+10,:,:], (1,2,0))
    mid = np.transpose(gt[5+10:,:,:], (1,2,0))
    return kp, short ,mid

def show_input(input):
    colors = [(0,0,0.9),(0.9,0,0),(0.9,0,0.9),(0.9,0.9,0), (0.2,0.9,0.9)]
    img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0 = input

    kp0, short0, mid0 = split_gt(gt_c0)
    kp1, short1, mid1 = split_gt(gt_c1)
    kp2, short2, mid2 = split_gt(gt_c2)
    kp3, short3, mid3 = split_gt(gt_c3)

    img = (img+0.5)*255
    img = np.transpose(img.data.cpu().numpy(), (1,2,0))

    map_s0 = np.zeros((short0.shape[:2]), dtype=np.float64)
    for i_kp in range(5):
        offset = short0[:, :, i_kp * 2:i_kp * 2 + 2]
        map_s0 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    cv2.imshow('short0', np.uint8((map_s0 * 200)))

    # map_m0 = np.zeros((mid0.shape[:2]), dtype=np.float64)
    # for i_kp in range(20):
    #     offset = mid1[:, :, i_kp * 2:i_kp * 2 + 2]
    #     map_m0 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    # cv2.imshow('mid0', np.uint8((map_m0 * 200)))

    map_s1 = np.zeros((short1.shape[:2]), dtype=np.float64)
    for i_kp in range(5):
        offset = short1[:, :, i_kp * 2:i_kp * 2 + 2]
        map_s1 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    cv2.imshow('short1', np.uint8((map_s1 * 200)))

    # map_m1 = np.zeros((mid1.shape[:2]), dtype=np.float64)
    # for i_kp in range(20):
    #     offset = mid1[:, :, i_kp * 2:i_kp * 2 + 2]
    #     map_m1 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    # cv2.imshow('mid1', np.uint8((map_m1 * 200)))

    map_s2 = np.zeros((short2.shape[:2]), dtype=np.float64)
    for i_kp in range(5):
        offset = short2[:, :, i_kp * 2:i_kp * 2 + 2]
        map_s2 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    cv2.imshow('short2', np.uint8((map_s2 * 200)))

    # map_m2 = np.zeros((mid2.shape[:2]), dtype=np.float64)
    # for i_kp in range(20):
    #     offset = mid2[:, :, i_kp * 2:i_kp * 2 + 2]
    #     map_m2 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    #     cv2.imshow('mid2', np.uint8((map_m2 * 200)))

    map_s3 = np.zeros((short3.shape[:2]), dtype=np.float64)
    for i_kp in range(5):
        offset = short3[:, :, i_kp * 2:i_kp * 2 + 2]
        map_s3 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    cv2.imshow('short3', np.uint8((map_s3 * 200)))

    # map_m3 = np.zeros((mid3.shape[:2]), dtype=np.float64)
    # for i_kp in range(20):
    #     offset = mid3[:, :, i_kp * 2:i_kp * 2 + 2]
    #     map_m3 += np.sqrt(offset[:, :, 0] ** 2 + offset[:, :, 1] ** 2)
    # cv2.imshow('mid3', np.uint8((map_m3 * 200)))

    cv2.imshow('img', np.uint8(img))

    k = cv2.waitKey(0)
    if k & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
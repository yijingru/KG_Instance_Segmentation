import numpy as np
import config as cfg

def create_position_index(height, width):
    """
    create 512x640x2 pixel position indexes
    each position represents (x,y)
    """
    position_indexes = np.rollaxis(np.indices(dimensions=(width, height)), 0, 3).transpose((1,0,2))
    return position_indexes


def copy_with_border_check(map, center, disc, kp_circle):
    """
    kp_circle: 2*KP_RADIUS x 2*KP_RADIUS
    cropped_disc:  2*KP_RADIUS x 2*KP_RADIUS
    map: height x width x 2
    disc: height x width
    """
    h, w = disc.shape

    center = (int(center[0]) ,int(center[1]))   # (x,y) for each y_{j,k}

    pad_t = max(cfg.KP_RADIUS - center[1], 0)
    pad_l = max(cfg.KP_RADIUS - center[0], 0)
    pad_b = max(cfg.KP_RADIUS - (h-1 - center[1]), 0)
    pad_r = max(cfg.KP_RADIUS - (w-1 - center[0]), 0)

    y1 = center[1] - cfg.KP_RADIUS + pad_t
    y2 = center[1] + cfg.KP_RADIUS - pad_b + 1
    x1 = center[0] - cfg.KP_RADIUS + pad_l
    x2 = center[0] + cfg.KP_RADIUS - pad_r + 1

    mask = disc[y1:y2, x1:x2]
    cropped_kp_circle = kp_circle[pad_t:2*cfg.KP_RADIUS-pad_b+1, pad_l:2*cfg.KP_RADIUS-pad_r+1, :]
    temp_map = np.zeros_like(map)
    temp_map[y1:y2, x1:x2 :] = cropped_kp_circle
    temp_map[np.where(mask)==0,:] = 0.

    # map[y1:y2, x1:x2 :][cropped_disc, :] = cropped_kp_circle[cropped_disc, :]
    map[y1:y2, x1:x2 :]= temp_map[y1:y2, x1:x2 :]

    return map

def compute_short_offsets(bboxes, discs, height, width):
    """
    bboxes: num_insts x num_kps x 2
    discs:  num_insts x num_kps x height x width
    """
    num_insts, num_kps,_ = bboxes.shape
    x = np.tile(np.arange(cfg.KP_RADIUS, -cfg.KP_RADIUS-1, -1), [2 * cfg.KP_RADIUS + 1, 1])# [7,6,...,0,1,...,6,7]
    y = x.transpose()
    mask = np.expand_dims(np.sqrt(x * x + y * y) <= cfg.KP_RADIUS, axis=-1)
    kp_circle = np.stack([x, y], axis=-1) * mask    # 15 x 15 x 2

    offsets = np.zeros((height, width, 2*num_kps))
    for i in range(num_kps):
        for j in range(num_insts):
            offsets[:, :, 2*i: 2*i+2] = copy_with_border_check(offsets[:, :, 2*i: 2*i+2],
                                                               (bboxes[j, i, 0], bboxes[j, i, 1]),
                                                               discs[j,i,:,:].astype(np.int),
                                                               kp_circle)
    return offsets


def load_disc_masks(bboxes, height, width):
    num_insts, num_kps, _ = bboxes.shape
    disc_masks = np.zeros(shape=(num_insts, num_kps, height, width))
    position_idxs = create_position_index(height, width)
    for i in range(num_kps):
        insts_pt = bboxes[:,i,:]
        dists_pt = np.zeros(shape=(height, width, num_insts))
        for j in range(num_insts):
            dists_pt[:,:,j] = np.sqrt(np.square(insts_pt[j,:]-position_idxs).sum(axis=-1))
        if num_insts>0:
            insts_id = dists_pt.argmin(axis=-1)  # [0,num_insts]
        for j in range(num_insts):
            mask = np.logical_and(insts_id==j, dists_pt[:,:,j]<=cfg.KP_RADIUS)
            disc_masks[j,i,:,:] = mask
    return disc_masks

def load_kp_heats(disc_masks):
    num_insts, num_kps, height, width = disc_masks.shape
    kp_heats = np.zeros(shape=(num_kps, height, width))
    for i in range(num_kps):
        for j in range(num_insts):
            kp_heats[i,:,:][np.where(disc_masks[j,i,:,:]==1.)] = 1.
    return kp_heats


def compute_mid_offsets(bboxes, discs):
    """
    bboxes: num_insts x num_kps x 2
    discs:  num_insts x num_kps x height x width
    edge value represents the number in num_kps.
    """
    num_insts, num_kps, height, width = discs.shape
    offsets = np.zeros((height, width, 4 * cfg.NUM_EDGES))    # 4: bi-direction+(x,y) 2+2
    position_idxs = create_position_index(height, width)  # position_idxs: (512, 640, 2)
    for i, edge in enumerate((cfg.EDGES + [edge[::-1] for edge in cfg.EDGES])):
        for j in range(num_insts):
            rr, cc = np.where(discs[j, edge[0], :, :]==1)
            distance = [[bboxes[j, edge[1], 0], bboxes[j, edge[1], 1]]] - position_idxs[rr, cc, :]
            offsets[rr, cc, 2*i:2*i+2] = distance
    return offsets

def get_ground_truth(bboxes, height, width, num_kps):
    if len(bboxes) == 0:
        kp_heats = np.zeros(shape=(num_kps, height, width))
        short_offsets = np.zeros((height, width, 2*num_kps))
        mid_offsets = np.zeros((height, width, 4 * cfg.NUM_EDGES))
        return kp_heats, short_offsets, mid_offsets

    disc_masks = load_disc_masks(bboxes, height, width)
    kp_heats = load_kp_heats(disc_masks)
    short_offsets = compute_short_offsets(bboxes, disc_masks, height, width)
    mid_offsets = compute_mid_offsets(bboxes, disc_masks)
    return kp_heats, short_offsets, mid_offsets
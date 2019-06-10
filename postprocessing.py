import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import maximum_filter
import config as cfg


def create_position_index(height, width):
    """
    create 512x640x2 pixel position indexes
    each position represents (x,y)
    """
    position_indexes = np.rollaxis(np.indices(dimensions=(width, height)), 0, 3).transpose((1,0,2))
    return position_indexes

def accumulate_votes(votes, shape):
    # Hough Voting
    xs = votes[:,0]
    ys = votes[:,1]
    ps = votes[:,2]
    tl = [np.floor(ys).astype('int32'), np.floor(xs).astype('int32')]
    tr = [np.floor(ys).astype('int32'), np.ceil(xs).astype('int32')]
    bl = [np.ceil(ys).astype('int32'), np.floor(xs).astype('int32')]
    br = [np.ceil(ys).astype('int32'), np.ceil(xs).astype('int32')]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps*(1.-dx)*(1.-dy)
    tr_vals = ps*dx*(1.-dy)
    bl_vals = ps*dy*(1.-dx)
    br_vals = ps*dy*dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    heatmap = np.asarray(coo_matrix( (data[good_inds], (I[good_inds],J[good_inds])), shape=shape ).todense())
    return heatmap

def compute_heatmaps(kp_maps, short_offsets):
    """
    kp_maps:  height x width x 5
    short_offset: 10 x height x width x 10
    """
    heatmaps = []
    height, width, num_kps = kp_maps.shape
    idx = create_position_index(height, width)
    for i in range(num_kps):
        this_kp_map = kp_maps[ :, :, i:i+1]
        votes = idx + short_offsets[:,:, 2*i:2*i+2]
        votes = np.reshape(np.concatenate([votes, this_kp_map], axis=-1), (-1, 3))  # (327680, 3)
        heatmaps.append(accumulate_votes(votes, shape=(height,width)) / (np.pi*cfg.KP_RADIUS**2))

    return np.stack(heatmaps, axis=-1)


def get_keypoints(heatmaps, peak_thresh):
    height, width, num_kps = heatmaps.shape
    keypoints = []
    for i in range(num_kps):
        peaks = maximum_filter(heatmaps[:,:,i], footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmaps[:,:,i]
        peaks = zip(*np.nonzero(peaks))
        keypoints.extend([{'id': i, 'xy': np.array(peak[::-1]), 'conf': heatmaps[peak[0], peak[1], i]} for peak in peaks])
        keypoints = [kp for kp in keypoints if kp['conf'] > peak_thresh]
    return keypoints



def iterative_bfs(graph, start, path=[]):
    '''iterative breadth first search from start'''
    q=[(None,start)]
    visited = []
    while q:
        v=q.pop(0)
        if not v[1] in visited:
            visited.append(v[1])
            path=path+[v]
            q=q+[(v[1], w) for w in graph[v[1]]]
    return path

def group_skeletons(keypoints, mid_offsets, heatmaps):
    """
    keypoints: list of dict:  {'id': 0, 'xy': array([212,  82]), 'conf': 0.018296129713976542}
    midoffsets: (512, 640, 40)   height x with x num_edges
    heatmaps: height x width x num_kps
    """
    height,width,num_kps = heatmaps.shape
    keypoints.sort(key=(lambda kp: kp['conf']), reverse=True)
    skeletons = []
    dir_edges = cfg.EDGES + [edge[::-1] for edge in cfg.EDGES]

    skeleton_graph = {i: [] for i in range(num_kps)}
    for i in range(num_kps):
        for j in range(num_kps):
            if (i, j) in cfg.EDGES or (j, i) in cfg.EDGES:
                skeleton_graph[i].append(j)
                skeleton_graph[j].append(i)

    while len(keypoints) > 0:
        kp = keypoints.pop(0)
        if any([np.linalg.norm(kp['xy'] - s[kp['id'], :2]) <= 10 for s in skeletons]):
            continue
        this_skel = np.zeros((num_kps, 3))
        this_skel[kp['id'], :2] = kp['xy']
        this_skel[kp['id'], 2] = kp['conf']
        path = iterative_bfs(skeleton_graph, kp['id'])[1:]
        for edge in path:
            if this_skel[edge[0], 2] == 0:
                continue
            mid_idx = dir_edges.index(edge)
            offsets = mid_offsets[:, :, 2 * mid_idx:2 * mid_idx + 2]
            from_kp = tuple(np.round(this_skel[edge[0], :2]).astype('int32'))
            proposal = this_skel[edge[0], :2] + offsets[from_kp[1], from_kp[0], :]
            matches = [(i, keypoints[i]) for i in range(len(keypoints)) if keypoints[i]['id'] == edge[1]]
            matches = [match for match in matches if np.linalg.norm(proposal - match[1]['xy']) <= cfg.KP_RADIUS+1]
            if len(matches) == 0:
                continue
            matches.sort(key=lambda m: np.linalg.norm(m[1]['xy'] - proposal))
            to_kp = np.round(matches[0][1]['xy']).astype('int32')
            to_kp_conf = matches[0][1]['conf']
            keypoints.pop(matches[0][0])
            this_skel[edge[1], :2] = to_kp
            this_skel[edge[1], 2] = to_kp_conf

        skeletons.append(this_skel)

    return skeletons


def get_skeletons_and_masks(kp_maps, short_offsets, mid_offsets):
    """
    kp_maps: batch x 5 x height x width
    short_offset: batch x 10 x height x width
    """
    kp_maps = kp_maps.data.cpu().numpy()
    short_offsets = short_offsets.data.cpu().numpy()
    mid_offsets = mid_offsets.data.cpu().numpy()

    kp_maps = np.transpose(kp_maps[0,:,:,:], (1,2,0))
    short_offsets = np.transpose(short_offsets[0,:,:,:], (1,2,0))
    mid_offsets = np.transpose(mid_offsets[0,:,:,:], (1,2,0))

    heatmaps = compute_heatmaps(kp_maps, short_offsets)
    for i in range(heatmaps.shape[2]):
        heatmaps[:,:,i] = gaussian_filter(heatmaps[:,:,i], sigma=2)
    pred_kp = get_keypoints(heatmaps, peak_thresh=0.004)
    skeletons = group_skeletons(pred_kp, mid_offsets, heatmaps)
    return skeletons


def refine_skeleton(skeletons):
    out_skeleton = []
    for skeleton in skeletons:
        mask = skeleton[:,0]>0.
        if mask.sum()>=3:
            out_skeleton.append(skeleton)
        elif mask[[0,3]].sum()==2 or mask[[1,2]].sum()==2:
            out_skeleton.append(skeleton)

    return out_skeleton




def skeleton_to_box(skeletons, scale):
    """
    skeletons: list of num_kp x 3 (x,y,conf)
    """
    bboxes = []
    for skeleton in skeletons:
        skeleton[:,:2] *= scale
        tl = skeleton[0,:]
        tr = skeleton[1,:]
        bl = skeleton[2,:]
        br = skeleton[3,:]
        cc = skeleton[4,:]
        mask = skeleton[:,0]>0.

        if mask[[0,1,2,3]].sum()==4:              # condition1: 4 corners exist
            y1 = min(tl[1], tr[1])
            y2 = max(bl[1], br[1])
            x1 = min(tl[0], bl[0])
            x2 = max(tr[0], br[0])
            conf = skeleton[mask,2].mean()
            bboxes.append([y1, x1, y2, x2, conf])

        elif mask[[0,1,2,3]].sum()==3:            # condition2: 3 corners exist
            y1 = np.where(mask[[0,1]].sum()==2, min(tl[1], tr[1]), max(tl[1], tr[1]))
            y2 = max(bl[1], br[1])
            x1 = np.where(mask[[0,2]].sum()==2, min(tl[0], bl[0]), max(tl[0], bl[0]))
            x2 = max(tr[0], br[0])
            conf = skeleton[mask,2].mean()
            bboxes.append([y1, x1, y2, x2, conf])

        elif mask[[0,1,2,3]].sum()==2:
            if mask[[0,3]].sum()==2:
                y1 = tl[1]
                y2 = br[1]
                x1 = tl[0]
                x2 = br[0]
                conf = skeleton[mask,2].mean()
                bboxes.append([y1, x1, y2, x2, conf])
            elif mask[[1,2]].sum()==2:
                y1 = tr[1]
                y2 = bl[1]
                x1 = bl[0]
                x2 = tr[0]
                conf = skeleton[mask,2].mean()
                bboxes.append([y1, x1, y2, x2, conf])


            elif mask[[0,1,4]].sum()==3:
                y1 = min(tl[1], tr[1])
                y2 = y1+(cc[1]-y1)*2
                x1 = tl[0]
                x2 = tr[0]
                conf = skeleton[mask,2].mean()
                bboxes.append([y1, x1, y2, x2, conf])

            elif mask[[0, 2, 4]].sum() == 3:
                y1 = tl[1]
                y2 = bl[1]
                x1 = min(tl[0], bl[0])
                x2 = x1 + (cc[0]-x1)*2
                conf = skeleton[mask, 2].mean()
                bboxes.append([y1, x1, y2, x2, conf])

            elif mask[[1, 3, 4]].sum() == 3:
                y1 = tr[1]
                y2 = br[1]
                x2 = max(tr[0], br[0])
                x1 = x2 - (x2-cc[0])*2
                conf = skeleton[mask, 2].mean()
                bboxes.append([y1, x1, y2, x2, conf])

            elif mask[[2, 3, 4]].sum() == 3:
                y2 = max(bl[1],br[1])
                y1 = y2- (y2-cc[1])*2
                x1 = bl[0]
                x2 = br[0]
                conf = skeleton[mask, 2].mean()
                bboxes.append([y1, x1, y2, x2, conf])
    return bboxes


def gather_skeleton_single(skeleton0, skeleton1, skeleton2, skeleton3):
    bboxes0 = skeleton_to_box(skeleton0, scale=1)
    bboxes1 = skeleton_to_box(skeleton1, scale=2)
    bboxes2 = skeleton_to_box(skeleton2, scale=4)
    bboxes3 = skeleton_to_box(skeleton3, scale=8)
    # bboxes = np.asarray(bboxes0+bboxes1+bboxes2+bboxes3)
    # return bboxes
    return np.asarray(bboxes0), np.asarray(bboxes1), np.asarray(bboxes2), np.asarray(bboxes3)


def gather_skeleton(skeleton0, skeleton1, skeleton2, skeleton3):
    bboxes0 = skeleton_to_box(skeleton0, scale=1)
    bboxes1 = skeleton_to_box(skeleton1, scale=2)
    bboxes2 = skeleton_to_box(skeleton2, scale=4)
    bboxes3 = skeleton_to_box(skeleton3, scale=8)
    bboxes = np.asarray(bboxes0+bboxes1+bboxes2+bboxes3)
    return bboxes
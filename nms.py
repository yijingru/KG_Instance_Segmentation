import os
import numpy as np

def non_maximum_suppression_numpy(bboxes, nms_thresh=0.5):
    """
    bboxes: num_insts x 5 [y1,y2,x1,x2,conf]
    """
    if len(bboxes)==0:
        return None
    y1 = bboxes[:,0]
    x1 = bboxes[:,1]
    y2 = bboxes[:,2]
    x2 = bboxes[:,3]
    conf = bboxes[:,4]
    area_all = (x2-x1)*(y2-y1)
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []

    while len(sorted_index)>0:
        # get the last biggest values
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        # pop the value
        sorted_index = sorted_index[:-1]
        # get the remaining boxes
        yy1 = np.take(y1, indices=sorted_index)
        xx1 = np.take(x1, indices=sorted_index)
        yy2 = np.take(y2, indices=sorted_index)
        xx2 = np.take(x2, indices=sorted_index)
        # get the intersection box
        yy1 = np.maximum(yy1, y1[curr_index])
        xx1 = np.maximum(xx1, x1[curr_index])
        yy2 = np.minimum(yy2, y2[curr_index])
        xx2 = np.minimum(xx2, x2[curr_index])
        # calculate IoU
        w = xx2-xx1
        h = yy2-yy1

        w = np.maximum(0., w)
        h = np.maximum(0., h)

        inter = w*h

        rem_areas = np.take(area_all, indices=sorted_index)
        union = (rem_areas-inter)+area_all[curr_index]
        IoU = inter/union
        sorted_index = sorted_index[IoU<=nms_thresh]

    out_bboxes = np.take(bboxes, keep_index, axis=0)

    return out_bboxes
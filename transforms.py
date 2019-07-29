import numpy as np
from numpy import random
import cv2




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ConvertImgFloat(object):
    def __call__(self, img, mask):
        return img.astype(np.float32), mask.astype(np.float32)

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, mask):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, mask


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, mask):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, mask

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, mask):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, mask


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, mask):
        img, mask = self.rb(img, mask)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img, mask = distort(img, mask)
        img, mask = self.rln(img, mask)
        return img, mask


class Expand(object):
    def __init__(self, max_scale = 2, mean = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, mask):
        if random.randint(2):
            return img, mask
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)

        expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
        expand_img[:,:,:] = self.mean
        expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img

        expand_mask = np.zeros(shape=(mask.shape[0], int(h*ratio), int(w*ratio)),dtype=mask.dtype)
        expand_mask[:, int(y1):int(y1+h), int(x1):int(x1+w)] = mask

        return expand_img, expand_mask


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win = 0.9):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            # (0.1, None),
            # (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, mask):
        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, mask

            for _ in range(50):
                current_img = img
                current_mask = mask

                w = random.uniform(self.min_win*width, width)
                h = random.uniform(self.min_win*height, height)
                if h/w<self.ratio[0] or h/w>self.ratio[1]:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                current_mask = current_mask[:, rect[0]:rect[2], rect[1]:rect[3]]

                return current_img, current_mask

class RandomMirror_w(object):
    def __call__(self, img, mask):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1,:]
            mask = mask[:,:,::-1]
        return img, mask

class RandomMirror_h(object):
    def __call__(self, img, mask):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:,:]
            mask = mask[:,::-1,:]
        return img, mask


class Resize(object):
    def __init__(self, h, w):
        self.dsize = (w,h)

    def __call__(self, img, mask):
        img = cv2.resize(img, dsize=self.dsize)
        mask_new = []
        for i in range(mask.shape[0]):
            cur_mask = mask[i,:,:]
            cur_mask = cv2.resize(cur_mask,dsize=self.dsize,interpolation=cv2.INTER_NEAREST)
            mask_new.append(cur_mask)
        return img, np.asarray(mask_new)
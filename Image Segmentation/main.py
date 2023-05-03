import timeit
import numpy as np
from skimage.draw import polygon

import paramparse

from predict import segmentate

def compute_seg(pred, gt):
    # pred value should be from 0 to 10, where 10 is the background.
    # accuracy is calculated for only non background pixels.
    assert pred.shape == gt.shape

    mask = gt != 10
    return (pred[mask] == gt[mask]).astype(int).sum() / gt[mask].size

class A8_Params:
    def __init__(self):
        self.prefix = "test"
        # self.prefix = "valid"
        # self.prefix = "train"
        self.vis = 0
        self.vis_size = (300, 300)
        self.show_pred = 1

        self.speed_thresh = 10
        self.seg_thresh = (0.7, 0.98)


def compute_score(res, thresh):
    min_thres, max_thres = thresh

    if res < min_thres:
        score = 0.0
    elif res > max_thres:
        score = 100.0
    else:
        score = float(res - min_thres) / (max_thres - min_thres) * 100
    return score


def main():
    params = A8_Params()
    paramparse.process(params)

    prefix = params.prefix

    images = np.load(prefix + "_X.npy")
    gt_segs = np.load(prefix + "_seg.npy")

    n_images = images.shape[0]

    print(f'running on {n_images} {prefix} images')

    start_t = timeit.default_timer()
    pred_segs = segmentate(images)
    end_t = timeit.default_timer()
    test_time = end_t - start_t

    assert test_time > 0, "test_time cannot be 0"

    test_speed = float(n_images) / test_time

    seg = compute_seg(pred_segs, gt_segs)

    seg_score = compute_score(seg, params.seg_thresh)

    if test_speed < params.speed_thresh:
        overall_score = 0
    else:
        overall_score = seg_score

    print(f"Segmentation Accuracy: {seg:.3f}")
    print(f"Test time: {test_time:.3f} seconds")
    print(f"Test speed: {test_speed:.3f} images / second")

    print(f"Overall Score: {overall_score:.3f}")

if __name__ == '__main__':
    main()

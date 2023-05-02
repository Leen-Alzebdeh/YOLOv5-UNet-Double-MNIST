from A7_submission import classify_and_detect
from skimage.draw import polygon
import numpy as np
import timeit
import os
import shutil

#from google.colab import drive
#drive.mount('/content/drive/', force_remount=True)
#!unzip -q content/MNISTDD_RGB_train_valid.zip -d content/
# os.system("git clone https://github.com/ultralytics/yolov5")


def resize_ar(src_img, width=0, height=0, return_factors=False,
              placement_type=0):
    '''
        Resize source image array to be the given height and width (300x300)
        :param src_img: source image array
        :param width: width to resize to
        :param height: height to resize to
        :return: iou score
    '''    
    import cv2

    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type == 0:
            start_row = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type == 0:
            start_col = 0
        elif placement_type == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)

    dst_img[start_row:start_row + src_height,
            start_col:start_col + src_width, :] = src_img
    dst_img = cv2.resize(dst_img, (width, height))
    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


def compute_classification_acc(pred, gt):
    '''
    Compute classification accuracy
    '''
    assert pred.shape == gt.shape
    return (pred == gt).astype(int).sum() / gt.size

def compute_iou(b_pred, b_gt):
    """
    Compute iou score

    :param b_pred: predicted bounding boxes, shape=(n,2,4)
    :param b_gt: ground truth bounding boxes, shape=(n,2,4)
    :return: iou score
    """

    n = np.shape(b_gt)[0]
    L_pred = np.zeros((64, 64))
    L_gt = np.zeros((64, 64))
    iou = 0.0
    for i in range(n):
        for b in range(2):
            rr, cc = polygon([b_pred[i, b, 0], b_pred[i, b, 0], b_pred[i, b, 2], b_pred[i, b, 2]],
                             [b_pred[i, b, 1], b_pred[i, b, 3], b_pred[i, b, 3], b_pred[i, b, 1]], [64, 64])
            L_pred[rr, cc] = 1

            rr, cc = polygon([b_gt[i, b, 0], b_gt[i, b, 0], b_gt[i, b, 2], b_gt[i, b, 2]],
                             [b_gt[i, b, 1], b_gt[i, b, 3], b_gt[i, b, 3], b_gt[i, b, 1]], [64, 64])
            L_gt[rr, cc] = 1

            iou += (1.0 / (2 * n)) * (np.sum((L_pred + L_gt) == 2) /
                                      np.sum((L_pred + L_gt) >= 1))

            L_pred[:, :] = 0
            L_gt[:, :] = 0

    return iou


class A7_Params:
    def __init__(self):
        # Select one 
        self.prefix = "test"
        # self.prefix = "valid"
        # self.prefix = "train"
        self.vis = 0
        self.vis_size = (300, 300)
        self.show_pred = 1

        self.speed_thresh = 10
        self.acc_thresh = (0.7, 0.98)
        self.iou_thresh = (0.7, 0.98)


def compute_score(res, thresh):
    min_thres, max_thres = thresh

    if res < min_thres:
        score = 0.0
    elif res > max_thres:
        score = 100.0
    else:
        score = float(res - min_thres) / (max_thres - min_thres) * 100
    return score


def draw_bboxes(img, bbox_1, bbox_2, y1, y2, vis_size):
    import cv2

    ymin, xmin, ymax, xmax = bbox_1

    cv2.rectangle(img, (int(xmin), int(ymin)), (int(
        xmax), int(ymax)), (0, 255, 0), thickness=1)
    cv2.putText(img, '{:d}'.format(y1), (xmin, ymin),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (0, 255, 0))

    ymin, xmin, ymax, xmax = bbox_2
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(
        xmax), int(ymax)), (255, 0, 0), thickness=1)
    cv2.putText(img, '{:d}'.format(y2), (xmin, ymin),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0))

    img = resize_ar(img, *vis_size)

    return img


def main():
    #!python train.py --img 640 --batch 16 --epochs 10 --data ../MNISTDD_RGB_train_valid/data.yaml --weights yolov5s.pt --cache --device 0
    params = A7_Params()
    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    prefix = params.prefix
    
    # load images, ground truth classes, ground truth boxes
    images = np.load(prefix + "_X.npy")
    gt_classes = np.load(prefix + "_Y.npy")
    gt_bboxes = np.load(prefix + "_bboxes.npy")

    os.chdir('yolov5')

    import torch
    from PIL import Image
    images = images.tolist()
    images = [np.array(i, dtype='uint8').reshape(i, (64, 64, 3))  for i in images]  # convert numpy array of images to a list of numpy array of images
    im = Image.fromarray(images[0], mode='RGB')
    im.save("image.jpg")  # visualize


    
    # load the model from the best parameters
    model = torch.hub.load("ultralytics/yolov5", 'custom',
                           path='../MNISTDD_RGB_train_valid/best.pt', _verbose=False)
    # Inference

    # sanity check
    # for i in images:
    #     results = model(i)
    #     results.print()

    os.chdir('..')
    n_images = images.shape[0]

    print(f'running on {n_images} {prefix} images')

    # check run time
    start_t = timeit.default_timer()

    pred_classes, pred_bboxes = classify_and_detect(images)

    end_t = timeit.default_timer()
    test_time = end_t - start_t

    assert test_time > 0, "test_time cannot be 0"

    test_speed = float(n_images) / test_time

    # compute performance
    acc = compute_classification_acc(pred_classes, gt_classes)
    iou = compute_iou(pred_bboxes, gt_bboxes)

    acc_score = compute_score(acc, params.acc_thresh)
    iou_score = compute_score(iou, params.iou_thresh)

    if test_speed < params.speed_thresh:
        overall_score = 0
    else:
        overall_score = (iou_score + acc_score) / 2

    print(f"Classification Accuracy: {acc:.3f}")
    print(f"Detection IOU: {iou:.3f}")
    print(f"Test time: {test_time:.3f} seconds")
    print(f"Test speed: {test_speed:.3f} images / second")

    print(f"Classification Score: {acc_score:.3f}")
    print(f"IOU Score: {iou_score:.3f}")
    print(f"Overall Score: {overall_score:.3f}")

    # to visualize 
    if params.vis:
        import cv2
        print('press space to taggle pause after each frame and escape to quit')
        pause_after_frame = 1
        for img_id in range(n_images):
            src_img = images[img_id, ...].squeeze().reshape(
                (64, 64, 3)).astype(np.uint8)
            vis_img = np.copy(src_img)
            vis_img_det = None

            if params.show_pred:
                vis_img_det = np.copy(src_img)

            bbox_1 = gt_bboxes[img_id, 0, :].squeeze().astype(np.int32)
            bbox_2 = gt_bboxes[img_id, 1, :].squeeze().astype(np.int32)
            y1, y2 = gt_classes[img_id, ...].squeeze()
            gt_classes[img_id, ...].squeeze()
            vis_img = draw_bboxes(vis_img, bbox_1, bbox_2,
                                  y1, y2, params.vis_size)

            if params.show_pred:
                bbox_1 = pred_bboxes[img_id, 0, :].squeeze().astype(np.int32)
                bbox_2 = pred_bboxes[img_id, 1, :].squeeze().astype(np.int32)
                y1, y2 = pred_classes[img_id, ...].squeeze()
                gt_classes[img_id, ...].squeeze()
                vis_img_det = draw_bboxes(
                    vis_img_det, bbox_1, bbox_2, y1, y2, params.vis_size)

                vis_img = np.concatenate((vis_img, vis_img_det), axis=1)

            cv2.imshow('vis_img', vis_img)

            key = cv2.waitKey(1 - pause_after_frame)
            if key == 27:
                return
            elif key == 32:
                pause_after_frame = 1 - pause_after_frame


if __name__ == '__main__':
    main()


# Scores
# Classification Score: 98.786
# IOU Score: 63.371
# Overall Score: 81.078
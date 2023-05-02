import numpy as np
import torch
import os
import shutil


def limit_preds(result):
    '''
    limit bounding box 
    '''
    max_conf = [0, 0]
    select_preds = torch.zeros(2, 6)  # result for both classes
    for preds in result[0]:
        if preds[4] > max_conf[0]:
            max_conf[0] = preds[4]
            select_preds[0] = preds
        elif preds[4] > max_conf[1]:
            max_conf[1] = preds[4]
            select_preds[1] = preds
    return select_preds


def generate_preds(pred_class, pred_bboxes, result, i):
    '''
    generate prediction bounding boxes and prediction classes
    :param np.ndarray array to load the predicted classes
    :param np.ndarray pred_bboxes: N x 12288 array containing N 64x64x3 images flattened into vectors
    :param np.ndarray result: N x 12288 array containing N 64x64x3 images flattened into vectors
    :param np.ndarray i: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: 
    '''
    pred = np.array([result[0][5], result[1][5]], dtype=np.int32)
    ind = np.argsort(pred)
    pred = pred[ind]
    pred_class[i] = pred
    pred_box = generate_pred_bboxes(result, i)
    pred_box = pred_box[ind]
    pred_bboxes[i] = pred_box

def generate_pred_bboxes(result, i):
    pred_box = np.empty((2, 4), dtype=np.int32)
    # [𝑦_𝑚𝑖𝑛,𝑥_𝑚𝑖𝑛,𝑦_𝑚𝑎𝑥,𝑥_𝑚𝑎𝑥]
    pred_box[0] = np.array(
        [result[0][1], result[0][0], result[0][3], result[0][2]], dtype=np.int32)
    pred_box[1] = np.array(
        [result[1][1], result[1][0], result[1][3], result[1][2]], dtype=np.int32)
    return pred_box

def classify_and_detect(images):
    """
    classify images and detect with bounding boxes
    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: prediction classes (np.ndarray), prediction boxes (np.ndarray)
    """
    os.system("git clone https://github.com/ultralytics/yolov5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir('yolov5')

    os.system('pip install -r requirements.txt  # install')

    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # Inference
    images = images.tolist()
    images = [np.array(i, dtype='uint8') for i in images]
    images = [np.reshape(i, (64, 64, 3)) for i in images]

    model = model = torch.hub.load(
        "ultralytics/yolov5", 'custom', path="../best.pt", force_reload=True, _verbose=False)
    model = model.to(device)
    for idx, image in enumerate(images):
        result = model(image, 460) # result from model prediction 
        ''' example of result:
                xmin    ymin    xmax   ymax  confidence  class    name
            tensor([[ 5.17117, 30.32191, 35.03222, 59.92870,  0.95923,  2.00000],
            [ 7.57471, 10.83041, 38.20247, 41.37091,  0.94121,  1.00000]])
        '''
        result = limit_preds(result.pred)
        generate_preds(pred_class, pred_bboxes, result, idx)

    os.chdir('..')
    # Results
    # results.print()
    return pred_class, pred_bboxes

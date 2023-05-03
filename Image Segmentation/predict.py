import numpy as np
import torch
import os
import shutil


def segmentate(images):
    """
​
    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = images.shape[0]

    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)
    os.chdir("unet-multi-seg/unet")
    from unet_model import UNet
    os.chdir("../checkpoints")
    model = UNet(3, 11).to(device)
    model.load_state_dict(torch.load(
        "checkpoint_epoch1.pth", map_location=device), strict=False)

    images = torch.tensor(images)

    images = images.astype(float)

    for i in range(N):
        images[i] = torch.reshape(images[i], (1, 64, 64, 3))
        images[i] = images.permute((0, 3, 1, 2))
        pred_seg[i] = model(images[i]).argmax(dim=1).flatten().cpu()

    # add your code here to fill in pred_seg
    return pred_seg
### CMPUT Course Project

### Author: Leen Alzebdeh

## Summary

I customize YOLOv5 and U-Net on a MNIST Double Digits RGB (MNISTDD-RGB) for a train-valid-test split dataset which was provided from the course, more details below.

## Object Detection on MNIST Double Digits RGB (MNISTDD-RGB)
Project page: [https://leen-alzebdeh.github.io/projects/328_detection/](https://leen-alzebdeh.github.io/projects/328_detection/)

Dataset consists of: 
- input: numpy array of numpy arrays which each represent pixels in the image, shape: number of samples, 12288 (flattened 64x64x3 images)
- output: 
  - classes: numpy array of numpy arrays which each represents the classes in the corresponding image, shape: number of samples, 2
  - prediction boxes: numpy array of numpy arrays which each represents the bounding boxes in the corresponding image, format: [y_min, x_min, y_max, x_max], shape: number of samples, 2, 4
  
I use YOLOv5 for object detection. I achieve a classification score of 98.786% and an IOU score of 63.371%, resulting in an overall score of 81.078%.

## Semantic Image Segmentation on MNIST Double Digits RGB (MNISTDD-RGB)
Project page: [https://leen-alzebdeh.github.io/projects/328_segmentation/](https://leen-alzebdeh.github.io/projects/328_segmentation/)

Dataset consists of: 
- input: numpy array of numpy arrays which each represent pixels in the image, shape: number of samples, 12288 (flattened 64x64x3 images)
- output: 
  - segementations: numpy array of numpy arrays which each represents the labels in the corresponding image, shape: number of samples, 4096 (flattened 64x64)

I customized a U-Net model for image segmentation. I achieve an accuracy of 87%.

## References

Pytorch. PyTorch. (n.d.). Retrieved May 2, 2023, from [https://pytorch.org/hub/ultralytics_yolov5/](https://pytorch.org/hub/ultralytics_yolov5/)

Kathuria, A. (2023, April 10). How to train Yolo V5 on a custom dataset. Paperspace Blog. Retrieved May 2, 2023, from [https://blog.paperspace.com/train-yolov5-custom-data/](https://blog.paperspace.com/train-yolov5-custom-data/)

Solawetz, J. (2020, September 29). How to train a custom object detection model with Yolo V5. Medium. Retrieved May 2, 2023, from [https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208 ](https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208)

(2017).Pytorch-Unet, from [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

I used the course's code templates from:

- Assignment 7: Object Detection/predict.py from A7_submission and Object detection/predict.py from A7_main
- Assignment 8: Image Segmentation/predict.py from A8_submission and Image Segmentation/predict.py from A8_main

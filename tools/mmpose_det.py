'''
Author: your name
Date: 2021-09-19 14:15:41
LastEditTime: 2021-09-19 14:48:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PoseDetectClassifier/mmpose_det.py
'''
import cv2
from torchvision.transforms import functional as F

from utils.transforms import get_affine_transform

img = None


# body detection
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(1333, 800)
_h, _w = img_scale
# scale and keep the scale ratio
resize_img = cv2.resize(img, (_h, _w))
transforms=[
    dict(type='Resize', keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
]



# pose detection
image_size = (192, 256)
center = (0)
scale  = 1
r      = 1
# TopDownAffine
trans = get_affine_transform(center, scale, r, image_size)
img = cv2.warpAffine(
    img,
    trans, (int(image_size[0]), int(image_size[1])),
    flags=cv2.INTER_LINEAR)


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
img[0] = (img[0] - mean[0])/std[0]
img[1] = (img[1] - mean[1])/std[1]
img[2] = (img[2] - mean[2])/std[2]

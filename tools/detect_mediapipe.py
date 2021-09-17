'''
Author: Baoyun Peng
Date: 2021-09-11 17:21:37
Description: using mediapipe to detect the pose of input image list
'''
# coding: utf-8
import cv2
import numpy as np
import random
import json
import os
import argparse

augment_type = [
    'stretch', 'flip', 'blur', 'noise'
]

def cv2_augmentation(img):
    height = img.shape[0]
    width = img.shape[1]
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.6:
        stretch_w = 1 + (random.random()-0.5)*0.25
        stretch_h = 1 + (random.random()-0.5)*0.25
        img = cv2.resize(img, (int(width*stretch_w), int(height*stretch_h)))
    if random.random() < 0.6:
        angle = random.uniform(25, 25)
        matRotate = cv2.getRotationMatrix2D((height // 2, width // 2), angle, 1)
        img = cv2.warpAffine(img, matRotate, (height, width))
    return img

def detect(prefix, imglist, aug_time=20):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    detect_result = {}

    with mp_pose.Pose(
        static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
    ) as pose:
        for line in open(imglist).readlines():
            img_path = line.split(' ')[0].strip()
            print(os.path.join(prefix, img_path))
            image = cv2.imread(os.path.join(prefix, img_path))
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            for aug_idx in range(aug_time):
                img = cv2_augmentation(image.copy())
                results = pose.process(img)
                if not results.pose_landmarks:
                    continue
                landmarks = np.array(
                    [
                        [
                            lmk.x * image.shape[1],
                            lmk.y * image.shape[0],
                            lmk.z * image.shape[1],
                            lmk.visibility,
                        ]
                        for lmk in results.pose_landmarks.landmark
                    ],
                    dtype=np.float32,
                )
                landmarks = np.round(landmarks, 5)
                _key = img_path.split('.')[0] + '_' + str(aug_idx) + '.' + img_path.split('.')[1]
                detect_result[_key] = landmarks.reshape(-1).tolist()
    return detect_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect 3D Human Pose')
    parser.add_argument('--imglist', type=str, help='specify the path of image list')
    parser.add_argument('--prefix', type=str, help='specify the prefix of image')
    parser.add_argument('--output', type=str, default="")
    args = parser.parse_args()
    detect_result = detect(args.prefix, args.imglist)
    if not args.output:
        output_path = os.path.splitext(args.imglist)[0] + '.json'
    else:
        output_path = args.output
    json.dump(detect_result, open(output_path, 'w'), indent=2)
    print(f'output path: {output_path}')


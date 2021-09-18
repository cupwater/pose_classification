'''
Author: Baoyun Peng
Date: 2021-09-18 12:29:58
Description: classify the pose and expression
'''

import cv2
import numpy as np
import json
import os
import argparse
import pdb
import torch
import mediapipe as mp

from models.WideResnet import wideresnet28_2
from models.fc_model import fc_A
from core.PoseEmbedding import FullBodyPoseEmbedder
from utils.align_face import warp_and_crop_face

# def int_tuple(t):
#     return tuple(int(x) for x in t)

def detect(pose, image):
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)
    if not results.pose_landmarks:
        return None
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
    return landmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='classify pose and expression')
    args = parser.parse_args()

    img_prefix = 'data/test_images/'
    # expr_list = ['happy.jpg', 'surprised.jpg']
    # expr_list = [os.path.join(img_prefix, 'expression', line) for line in expr_list ]
    pose_list = ['lean.jpg', 'lopsided.jpg', 'chin.jpg', 'down.jpg', 'normal.jpg' ]
    pose_list = [os.path.join(img_prefix, 'pose', line) for line in pose_list ]

    expr_model = wideresnet28_2(n_class=7)
    expr_model.load_state_dict(torch.load('weights/expression.npy', map_location='cpu')['model'])
    # we concat the part landmarks and its embedding into a 139-d feature vector
    pose_model = fc_A(139, 5)

    # Get given pose embedding.
    pose_embedding = FullBodyPoseEmbedder()    
    # pdb.set_trace()
    # return np.array(pose_embedding, dtype=np.float32), self.labels[idx]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) 

    for img_path in pose_list:
        image = cv2.imread(img_path.strip())
        landmarks = detect(pose, image)
        if landmarks is not None:
            
            src_pts = np.int32(landmarks[[5,2,0,9,10]][:, :2])
            src_pts = [ (image.shape[1] - pts[0], pts[1]) for pts in src_pts ]
            # for pts in src_pts:
            #     cv2.circle(image, int_tuple(pts), 2, (0,155,255), 2)
            # cv2.imshow('image.jpg', image)
            # key = cv2.waitKey(3)
            
            # crop the face region for expression recognition
            face_img = warp_and_crop_face(image, src_pts, align_type='similarity')
            face_img = np.transpose(face_img, (2,0,1))
            # cv2.imshow('image.jpg', face_img)
            # key = cv2.waitKey(-1)
            # if key == 27:
            #     continue
            inputs = torch.autograd.Variable(torch.from_numpy(face_img[np.newaxis,:,:,:]).float())
            predict_expr = expr_model(inputs)
            print(predict_expr)

            # get the embedding of pose landmarks
            inputs = np.array(pose_embedding(landmarks), dtype=np.float32)
            inputs = torch.autograd.Variable(torch.from_numpy(inputs[np.newaxis,:,]).float())
            predict_pose = pose_model(inputs)
            print(predict_pose)
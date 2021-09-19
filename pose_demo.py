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

from models.fc_model import fc_A
from core.PoseEmbedding import FullBodyPoseEmbedder
from utils.align_face import warp_and_crop_face


pose_dict = {
    '0': '托腮',
    '1': '低头',
    '2': '侧倾',
    '3': '高低肩',
    '4': '正常'
}


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
    img_list = ['lean.jpg', 'lopsided.jpg', 'chin.jpg', 'down.jpg', 'normal.jpg' ]
    img_list = [os.path.join(img_prefix, 'pose', line) for line in img_list ]

    # we concat the part landmarks and its embedding into a 139-d feature vector
    pose_model = fc_A(139, 5)
    pose_model.eval()
    pose_embedding = FullBodyPoseEmbedder()

    detect_results = {}

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) 
    for img_path in img_list:
        image = cv2.imread(img_path.strip())
        landmarks = detect(pose, image)
        face_img = None
        if landmarks is not None:
            src_pts = np.int32(landmarks[[2,5,0,9,10]][:, :2])
            src_pts = [ (image.shape[1] - pts[0], pts[1]) for pts in src_pts ]
            # crop the face region for expression recognition
            face_img = warp_and_crop_face(image, src_pts, align_type='affine')
            # get the embedding of pose landmarks
            inputs = np.array(pose_embedding(landmarks), dtype=np.float32)
            inputs = torch.autograd.Variable(torch.from_numpy(inputs[np.newaxis,:,]).float())
            predict_pose = pose_model(inputs)
            probs = torch.nn.functional.softmax(predict_pose)
            _pose = pose_dict[str(torch.argmax(predict_pose).item())]
            score = torch.max(probs).item()

            # get the bounding box from landmarks
            lx = int(np.min(landmarks[:, 0]))
            ly = int(np.min(landmarks[:, 1]) - (src_pts[4][0] - src_pts[0][0]))
            rx = int(np.max(landmarks[:, 0]))
            ry = int(np.max(landmarks[:, 1]))
    
            detect_results[img_path] = {
                "label": _pose,
                "score": score,
                "body_bbox": [lx, ly, rx, ry]
            }
    
    json.dump(detect_results, open('outputs/body.json', 'w'), ensure_ascii=False)

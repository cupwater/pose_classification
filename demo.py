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
from mtcnn import MTCNN

# expression_dict = {
#     '0': 'angry',
#     '1': 'disgusted',
#     '2': 'fearful',
#     '3': 'happy',
#     '4': 'neutral',
#     '5': 'sad',
#     '6': 'surprised'
# }
expression_dict = {
    '0': '生气',
    '1': '恶心',
    '2': '害怕',
    '3': '高兴',
    '4': '中性',
    '5': '悲伤',
    '6': '吃惊'
}

# pose_dict = {
#     '0': 'chin',
#     '1': 'down',
#     '2': 'lean',
#     '3': 'lopsided',
#     '4': 'normal'
# }
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
    img_list = ['happy.jpg', 'surprised.jpg']
    img_list1 = [os.path.join(img_prefix, 'expression', line) for line in img_list ]
    img_list = ['lean.jpg', 'lopsided.jpg', 'chin.jpg', 'down.jpg', 'normal.jpg' ]
    img_list2 = [os.path.join(img_prefix, 'pose', line) for line in img_list ]
    img_list = img_list1 + img_list2

    expr_model = wideresnet28_2(n_class=7)
    expr_model.load_state_dict(torch.load('weights/expression.npy', map_location='cpu')['model'])
    expr_model.eval()
    # we concat the part landmarks and its embedding into a 139-d feature vector
    pose_model = fc_A(139, 5)
    # pose_model.load_state_dict(torch.load('weights/expression.npy', map_location='cpu')['model'])
    pose_model.eval()
    pose_embedding = FullBodyPoseEmbedder()    

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) 
    detector = MTCNN()
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
            _pose = pose_dict[str(torch.argmax(predict_pose).item())]
            # print(f'pose of {img_path} is {_pose}')
        else:
            # use mtcnn when mediapipe fails to detect face and pose
            mtcnn_result = detector.detect_faces(image)
            if len(mtcnn_result) == 0:
                continue
            # get the biggest region as detect result
            max_area = 0
            _det_idx = 0
            for _idx, _det in enumerate(mtcnn_result):
                lx, ly, h, w = _det['box']
                if max_area < h*w:
                    max_area = h*w
                    _det_idx = _idx
            lx, ly, h, w = mtcnn_result[_det_idx]['box']
            face_img = image[ly:(ly+w), lx:(lx+h), : ]

        if face_img is not None:
            face_img = np.transpose(cv2.resize(face_img, (112, 112)), (2,0,1))
            face_img = (face_img / 255.0 - 0.508) / 0.255
            inputs = torch.autograd.Variable(torch.from_numpy(face_img[np.newaxis,:,:,:]).float())
            predict_expr = expr_model(inputs)
            _expr = expression_dict[str(torch.argmax(predict_expr).item())]
            print(f'expression of {img_path} is {_expr}')
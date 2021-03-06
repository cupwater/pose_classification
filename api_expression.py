'''
Author: Baoyun Peng
Date: 2021-09-18 12:29:58
Description: classify the pose and expression
'''

import cv2
import numpy as np
import torch

from models.WideResnet import wideresnet28_2
from mtcnn import MTCNN

__all__ = ['detect_expression']

expression_dict = {
    '0': '生气',
    '1': '恶心',
    '2': '害怕',
    '3': '高兴',
    '4': '中性',
    '5': '悲伤',
    '6': '吃惊'
}


def detect_expression(img_list):
    expr_model = wideresnet28_2(n_class=7)
    expr_model.load_state_dict(
        torch.load('weights/expression.npy', map_location='cpu')['model'])
    expr_model.eval()
    detector = MTCNN()
    detect_results = {}
    for img_path in img_list:
        image = cv2.imread(img_path.strip())
        detect_results[img_path] = []
        # use mtcnn when mediapipe fails to detect face and pose
        mtcnn_result = detector.detect_faces(image)
        if len(mtcnn_result) == 0:
            continue
        for _idx, _det in enumerate(mtcnn_result):
            lx, ly, h, w = _det['box']
            face_img = image[ly:(ly + w), lx:(lx + h), :]
            face_img = np.transpose(cv2.resize(face_img, (112, 112)),
                                    (2, 0, 1))
            face_img = (face_img / 255.0 - 0.508) / 0.255
            inputs = torch.autograd.Variable(
                torch.from_numpy(face_img[np.newaxis, :, :, :]).float())
            predict_expr = expr_model(inputs)
            probs = torch.nn.functional.softmax(predict_expr, dim=1)
            expr = expression_dict[str(torch.argmax(probs).item())]
            score = torch.max(probs).item()

            detect_results[img_path.strip()].append({
                "label": expr,
                "score": score,
                "face_bbox": [lx, ly, lx + h, ly + w]
            })
    return detect_results

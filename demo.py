"""
Author: Baoyun Peng
Date: 2021-09-11 21:28:03
Description: use pose landmarks to classify the pose type
"""

import os
import csv

from utils.bootstrap_helper import BootstrapHelper
from core.PoseEmbedding import FullBodyPoseEmbedder
from core.PoseClassifier import PoseClassifier



# Transforms pose landmarks into embedding.
pose_embedder = FullBodyPoseEmbedder()
# Classifies give pose against database of poses.
pose_classifier = PoseClassifier(
    landmarks_path='data/train.json',
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10,
)
outliers = pose_classifier.find_pose_sample_outliers()
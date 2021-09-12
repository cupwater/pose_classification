'''
Author: Baoyun Peng
Date: 2021-09-12
Description: pose dataset
'''
import csv
import numpy as np
import os
import json
import torch.utils.data as data

import pdb

class PoseSample(object):
    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


class PoseDataset(data.Dataset):
    def __init__(
        self,
        landmarks_path,
        pose_embedder,
        n_landmarks=33,
        n_dimensions=4,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
        axes_weights=(1.0, 1.0, 0.2),
    ):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(
            landmarks_path,
            n_landmarks,
            n_dimensions,
            pose_embedder,
        )


        class_list = [ sample.class_name for sample in self._pose_samples ]
        classes = list(set(class_list))
        self.labels = [ classes.index(name) for name in class_list ]
        

    def _load_pose_samples(
        self,
        landmarks_path,
        n_landmarks,
        n_dimensions,
        pose_embedder,
    ):
        """
        Loads pose samples from a json file
        """
        # Each file in the folder represents one pose class.
        landmarks_dict = json.load(open(landmarks_path, 'r'))

        pose_samples = []
        for key, landmarks in landmarks_dict.items():
            # Use file name as pose class name.
            class_name = key.split('/')[0]
            landmarks  = np.array(landmarks).reshape(n_landmarks, n_dimensions)
            pose_samples.append(
                PoseSample(
                    name=class_name,
                    landmarks = landmarks,
                    class_name=class_name,
                    embedding=pose_embedder(landmarks),
                )
            )

        return pose_samples


    def __getitem__(self, idx):
        """get embedding feature of a given pose landmark.
        """
        pose_landmarks = self._pose_samples[idx]
        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks.landmarks)
        # pdb.set_trace()
        return np.array(pose_embedding, dtype=np.float32), self.labels[idx]

    def __len__(self):
        return len(self._pose_samples)


if __name__ == "__main__":
    # Classifies give pose against database of poses.
    from core.PoseEmbedding import FullBodyPoseEmbedder
    pose_embedder = FullBodyPoseEmbedder()
    pose_classifier = PoseDataset(
        landmarks_path='data/train.json',
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)
    feature = pose_classifier.__getitem__(1)
    import pdb
    pdb.set_trace()

import csv
import numpy as np
import os
import json

class PoseSample(object):
    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


class PoseSampleOutlier(object):
    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(
        self,
        landmarks_path,
        pose_embedder,
        n_landmarks=33,
        n_dimensions=3,
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
        landmarks_dict = json.load(landmarks_path)

        pose_samples = []
        for key, landmarks in landmarks_dict:
            # Use file name as pose class name.
            class_name = key.split('/')[0]
            pose_samples.append(
                PoseSample(
                    name=class_name,
                    landmarks=landmarks.reshape(n_landmarks, n_dimensions),
                    class_name=class_name,
                    embedding=pose_embedder(landmarks),
                )
            )

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database."""
        # Find outliers in target poses
        outliers = []
        for sample in self._pose_samples:
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [
                class_name
                for class_name, count in pose_classification.items()
                if count == max(pose_classification.values())
            ]

            # Sample is an outlier if nearest poses have different class or more than
            # one pose class is detected as nearest.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(
                    PoseSampleOutlier(sample, class_names, pose_classification)
                )

        return outliers

    def __call__(self, pose_landmarks):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (
            self._n_landmarks,
            self._n_dimensions,
        ), "Unexpected shape: {}".format(pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(
            pose_landmarks * np.array([-1, 1, 1])
        )

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[: self._top_n_by_max_distance]

        # Filter by mean distance.
        #
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[: self._top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [
            self._pose_samples[sample_idx].class_name
            for _, sample_idx in mean_dist_heap
        ]
        result = {
            class_name: class_names.count(class_name) for class_name in set(class_names)
        }

        return result

'''
Author: Baoyun Peng
Date: 2021-09-18 12:29:58
Description: classify the pose and expression
'''

import cv2
import numpy as np
import torch
import os
import struct
import mediapipe as mp
# from utils.crypt import decrypt_file

from Crypto.Cipher import AES

try:
    from Crypto.Util.Padding import pad, unpad
except ImportError:
    from Crypto.Util.py3compat import bchr, bord
    def pad(data_to_pad, block_size):
        padding_len = block_size-len(data_to_pad)%block_size
        padding = bchr(padding_len)*padding_len
        return data_to_pad + padding
    def unpad(padded_data, block_size):
        pdata_len = len(padded_data)
        if pdata_len % block_size:
            raise ValueError("Input data is not padded")
        padding_len = bord(padded_data[-1])
        if padding_len<1 or padding_len>min(block_size, pdata_len):
            raise ValueError("Padding is incorrect.")
        if padded_data[-padding_len:]!=bchr(padding_len)*padding_len:
            raise ValueError("PKCS#7 padding is incorrect.")
        return padded_data[:-padding_len]

def decrypt_file(key, in_filename, out_filename=None, chunksize=64*1024):
    if not out_filename:
        out_filename = in_filename + '.dec'
    with open(in_filename, 'rb') as infile:
        filesize = struct.unpack('<Q', infile.read(8))[0]
        iv = infile.read(16)
        encryptor = AES.new(key, AES.MODE_CBC, iv)
        with open(out_filename, 'wb') as outfile:
            encrypted_filesize = os.path.getsize(in_filename)
            pos = 8 + 16 # the filesize and IV.
            while pos < encrypted_filesize:
                chunk = infile.read(chunksize)
                pos += len(chunk)
                chunk = encryptor.decrypt(chunk)
                if pos == encrypted_filesize:
                    chunk = unpad(chunk, AES.block_size)
                outfile.write(chunk) 

class fc_A(torch.nn.Module):
    def __init__(self, in_feature, class_num):
        super(fc_A, self).__init__()
        self.fc1 = torch.nn.Linear(in_feature, 256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, class_num)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky_1",
            "right_pinky_1",
            "left_index_1",
            "right_index_1",
            "left_thumb_2",
            "right_thumb_2",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

    def __call__(self, landmarks_visibility):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks_visibility - NumPy array with 3D landmarks and visibility of shape (N, 4).
        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        # import pdb
        # pdb.set_trace()
        # landmarks_visibility = landmarks_visibility
        assert landmarks_visibility.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(landmarks_visibility.shape[0])

        # Get pose landmarks.
        lmk3d = np.copy(landmarks_visibility[:, :3])
        # Normalize landmarks.
        lmk3d = self._normalize_pose_landmarks(lmk3d)
        # Get embedding.
        embedding = self._get_pose_distance_embedding(lmk3d).reshape(-1)

        lmk3d_visible = np.concatenate((lmk3d[:25, :], landmarks_visibility[:25, 3:]), axis=1).reshape(-1)
        return lmk3d_visible.tolist() + embedding.tolist()

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        # landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index("left_hip")]
        right_hip = landmarks[self._landmark_names.index("right_hip")]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index("left_shoulder")]
        right_shoulder = landmarks[self._landmark_names.index("right_shoulder")]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array(
            [
                # One joint.
                self._get_distance(
                    self._get_average_by_names(landmarks, "left_hip", "right_hip"),
                    self._get_average_by_names(
                        landmarks, "left_shoulder", "right_shoulder"
                    ),
                ),
                self._get_distance_by_names(landmarks, "left_shoulder", "left_elbow"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_elbow"),
                self._get_distance_by_names(landmarks, "left_elbow", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_elbow", "right_wrist"),
                # self._get_distance_by_names(landmarks, "left_hip", "left_knee"),
                # self._get_distance_by_names(landmarks, "right_hip", "right_knee"),
                # self._get_distance_by_names(landmarks, "left_knee", "left_ankle"),
                # self._get_distance_by_names(landmarks, "right_knee", "right_ankle"),
                # Two joints.
                self._get_distance_by_names(landmarks, "left_shoulder", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_wrist"),
                # self._get_distance_by_names(landmarks, "left_hip", "left_ankle"),
                # self._get_distance_by_names(landmarks, "right_hip", "right_ankle"),
                # Four joints.
                self._get_distance_by_names(landmarks, "left_hip", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_hip", "right_wrist"),
                # Five joints.
                # self._get_distance_by_names(landmarks, "left_shoulder", "left_ankle"),
                # self._get_distance_by_names(landmarks, "right_shoulder", "right_ankle"),
                self._get_distance_by_names(landmarks, "left_hip", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_hip", "right_wrist"),
                # Cross body.
                self._get_distance_by_names(landmarks, "left_elbow", "right_elbow"),
                # self._get_distance_by_names(landmarks, "left_knee", "right_knee"),
                self._get_distance_by_names(landmarks, "left_wrist", "right_wrist"),
                # self._get_distance_by_names(landmarks, "left_ankle", "right_ankle"),
                # Body bent direction.
                # self._get_distance(
                #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
                #     landmarks[self._landmark_names.index('left_hip')]),
                # self._get_distance(
                #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
                #     landmarks[self._landmark_names.index('right_hip')]),
            ]
        )

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

pose_dict = {
    '0': '侧倾',
    '1': '托腮',
    '2': '高低肩',
    '3': '低头',
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

def detect_pose(img_list):
    pose_model = fc_A(139, 5)
    decrypt_file("pby-pose-1234567".encode('utf-8'),"weights/pose.npy.enc", out_filename='/tmp/pose.npy')
    pose_model.load_state_dict(
            torch.load('/tmp/pose.npy', map_location='cpu'))
    pose_model.eval()
    pose_embedding = FullBodyPoseEmbedder()
    detect_results = {}
    os.system('rm /tmp/pose.npy')

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) 
    for img_path in img_list:
        image = cv2.imread(img_path.strip())
        landmarks = detect(pose, image)
        if landmarks is not None:
            src_pts = np.int32(landmarks[[2,5,0,9,10]][:, :2])
            src_pts = [ (image.shape[1] - pts[0], pts[1]) for pts in src_pts ]
            # get the embedding of pose landmarks
            inputs = np.array(pose_embedding(landmarks), dtype=np.float32)
            inputs = torch.autograd.Variable(torch.from_numpy(inputs[np.newaxis,:,]).float())
            predict_pose = pose_model(inputs)
            probs = torch.nn.functional.softmax(predict_pose, dim=1)
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
    return detect_results

'''
Author: Baoyun Peng
Date: 2021-09-13 
Description: use mtcnn to detect face
'''
from mtcnn import detect_faces, show_bboxes
from PIL import Image
import cv2

img = Image.open('data/lege/test1.jpg')
# bounding_boxes, landmarks = detect_faces(img)
# show_bboxes(img, bounding_boxes, landmarks)

# img = Image.open('data/images/office2.jpg')
# bounding_boxes, landmarks = detect_faces(img)
# show_bboxes(img, bounding_boxes, landmarks)

# img = Image.open('data/images/office3.jpg')
# bounding_boxes, landmarks = detect_faces(img)
# show_bboxes(img, bounding_boxes, landmarks)

# img = Image.open('data/images/office4.jpg')
bounding_boxes, landmarks = detect_faces(img, thresholds=[0.6, 0.7, 0.85])
img = show_bboxes(img, bounding_boxes, landmarks)

img.show()

# img = Image.open('data/images/office5.jpg')
# bounding_boxes, landmarks = detect_faces(img, min_face_size=10.0)
# show_bboxes(img, bounding_boxes, landmarks)
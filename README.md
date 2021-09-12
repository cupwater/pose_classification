# pose_classification
This Project is used to classify the pose of a given picture. The pipeline is as follows:
1. detect the pose landmarks;
2. obtains the embedding of pose by combining 3D landmarks and the pose distances
3. train the classifier on trainset, the input of classifier is embedding mentioned in step 2.
4. using classifier to classify a given pose

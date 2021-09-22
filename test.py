'''
Author: Baoyun Peng
Date: 2021-09-18 12:29:58
Description: demo for classifying pose and expression fir a given image list
'''
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='classify pose and expression')
    parser.add_argument('--image-list',
                        default="data/test_image.lst",
                        type=str,
                        help='image list')
    parser.add_argument(
        '--detect-type',
        default=0,
        type=int,
        help='specify the task type, 0 for pose, 1 for expression')

    args = parser.parse_args()
    imglist = open(args.image_list).readlines()

    if args.detect_type == 0:
        from api_pose import detect_pose
        detect_result = detect_pose(imglist)
        json.dump(detect_result, open('outputs/pose.json', 'w'), ensure_ascii=False)
    elif args.detect_type == 1:
        from api_expression import detect_expression
        detect_result = detect_expression(imglist)
        json.dump(detect_result, open('outputs/expression.json', 'w'), ensure_ascii=False)

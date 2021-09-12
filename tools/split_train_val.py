'''
Author: Baoyun Peng
Date: 2021-09-12 00:14:47
Description: split the input list into train and val list
'''

import random

split_ratio = 0.85

data_list = open('data/all_list.txt').readlines()
random.shuffle(data_list)
train_list = data_list[:int(len(data_list)*split_ratio)]
val_list = data_list[int(len(data_list)*split_ratio):]

with open('data/val.list', 'w') as val_out:
    val_out.writelines("".join(val_list))

with open('data/train.list', 'w') as train_out:
    train_out.writelines("".join(train_list))

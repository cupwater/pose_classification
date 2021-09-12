'''
Author: Baoyun
Date: 2021-09-12 
Description: model
'''
import torch.nn as nn

class fc_A(nn.Module):
    def __init__(self, in_feature, class_num):
        super(fc_A, self).__init__()
        self.fc1 = nn.Linear(in_feature, 256)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_feature, 128)
        self.sig2 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_feature, class_num)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sig1(x)
        x = self.fc2(x)
        x = self.sig2(x)
        x = self.fc3(x)
        return x
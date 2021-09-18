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
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, class_num)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class fc_S(nn.Module):
    def __init__(self, in_feature, class_num):
        super(fc_S, self).__init__()
        self.fc1 = nn.Linear(in_feature, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, class_num)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


'''
Author: Baoyun
Date: 2021-09-12 
Description: model
'''
import torch.nn as nn

class fc_B(nn.Module):
    def __init__(self, in_feature, class_num):
        super(fc_B, self).__init__()
        self.fc1 = nn.Linear(in_feature, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, class_num)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):   #
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)


        return x

import torch
input1 = torch.rand([32,1,32,32])    #第一个32，batch
model = LeNet()                         # 实例化模型
print(model)
outout = model(input1)
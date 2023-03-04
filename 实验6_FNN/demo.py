import torch
from torch import nn
linear = nn.Linear(32, 2)    #输入32维，输出2维
inputs = torch.rand(3,32)    #创建一个形状为(3,32)的随机张量 3为batch,32为in_features
outputs = linear(inputs)     #输出张量，大小为(3,2)
print(outputs)

from torch.nn import functional as F
activation = torch.sigmoid(outputs)
print(activation)

activation=F.softmax(outputs,dim=1)
print(activation)

activation = F.relu(outputs)
print(activation)


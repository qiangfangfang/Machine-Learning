# import torch
# from torch import nn
# linear = nn.Linear(32, 2) #输入32维，输出2维
# inputs = torch.rand(3,32) #创建一个形状为(3,32)的随机张量 3 为batch,32为in_features
# outputs = linear(inputs) #输出张量，大小为(3,2)
# print(outputs)
# #
# #
# from torch.nn import functional as F
# activation = F.sigmoid(outputs)
# print(activation)
# activation = torch.sigmoid(outputs)
# print(activation)
# #
# activation=F.softmax(outputs,dim=1)
# print(activation)
# #
# activation = F.relu(outputs)
# print(activation)
# #
# import torch
# from torch import nn
# from torch.nn import functional as F
# class MLP(nn.Module):
#       def __init__(self, input_dim, hidden_dim, num_class):
#        super(MLP, self).__init__()# 线性变换：输入层->隐含层
#        self.linear1 = nn.Linear(input_dim, hidden_dim)# 使用ReLU激活函数
#        self.activate = F.relu# 线性变换：隐含层->输出层
#        self.linear2 = nn.Linear(hidden_dim, num_class)
#       def forward(self, inputs):
#        hidden = self.linear1(inputs)
#        activation = self.activate(hidden)
#        outputs = self.linear2(activation)
#        probs = F.softmax(outputs, dim=1) # 获得每个输入属于某一类别的概率
#        return probs
# #
# mlp = MLP(input_dim=4, hidden_dim=5, num_class=2)
# inputs = torch.rand(3, 4)  # 输入形状为(3, 4)的张量，其中3表示有3 个输入，4 表示每个输入的维度
# probs = mlp(inputs)  # 自动调用forward函数
# print(probs)  # 输出3个输入对应输出的概率
# #


# import torch
# from torch import nn, optim
# from torch.nn import functional as F
# class MLP(nn.Module):
#  def __init__(self, input_dim, hidden_dim, num_class):
#        super(MLP, self).__init__()
#        self.linear1 = nn.Linear(input_dim, hidden_dim)
#        self.activate = F.relu
#        self.linear2 = nn.Linear(hidden_dim, num_class)
#  def forward(self, inputs):
#        hidden = self.linear1(inputs)
#        activation = self.activate(hidden)
#        outputs = self.linear2(activation)
#        log_probs = F.log_softmax(outputs, dim=1) #取对数：避免计算softmax时可能产生的数值溢出问题
#        return log_probs
# # 异或问题的4个输入
# x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# # # 每个输入对应的输出类别
# y_train = torch.tensor([0, 1, 1, 0])
# # # 创建多层感知器模型，输入层大小为2，隐含层大小为5，输出层大小为2（即有两个类别）
# model = MLP(input_dim=2, hidden_dim=5, num_class=2)
# criterion = nn.NLLLoss()
# # # 当使用log_softmax输出时，需要调用负对数似然损失（Negative LogLikelihood，NLL）
# optimizer = optim.SGD(model.parameters(), lr=0.05) # 使用梯度下降参数优化方法，学习率设置为0.05
# for epoch in range(100):
#      y_pred = model(x_train) # 调用模型，预测输出结果
#      loss = criterion(y_pred, y_train) # 通过对比预测结果与正确的结果，计算损失
#      optimizer.zero_grad() # 在调用反向传播算法之前，将优化器的梯度值置为零，否则每次循环的梯度将进行累加
#      loss.backward() # 通过反向传播计算参数的梯度
#      optimizer.step() # 在优化器中更新参数，不同优化器更新的方法不同，但是调用方式相同
# print("Parameters:")
# for name, param in model.named_parameters():
#     print (name, param.data)
# y_pred = model(x_train)
# print("Predicted results:", y_pred.argmax(axis=1))

#  MNIST数据集训练
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 读取数据
train_data = datasets.MNIST(root="./data", train=True,
                            download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="./data", train=False,
                           download=True, transform=transforms.ToTensor())

train_data, test_data
# 创建数据加载器
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
# 查看数据
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)
# 图像可视化
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("True Label: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# 模型定义
import torch.nn as nn
import torch.nn.functional as F

class MLP4(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_class):
        super(MLP4, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 200)
        self.linear3 = nn.Linear(200, 150)
        self.linear4 = nn.Linear(150, 100)
        self.linear5 = nn.Linear(100, num_class)

    def forward(self, data):
        #先将图片数据转化为1*784的张量
        data = data.view(-1, 28*28)
        data = F.relu(self.linear1(data))
        data = F.relu(self.linear2(data))
        data = F.relu(self.linear3(data))
        data = F.relu(self.linear4(data))
        data = self.linear5(data)
        data = F.log_softmax(data, dim = 1)
        return data


input_size = 28 * 28
hidden_size = 500
class_num = 10
epochs = 10

model = MLP4(input_size, hidden_size, class_num)
print(model)

import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
total_step = len(train_loader)

for epoch in range(epochs):
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

correct = 0
total = 0
with torch.no_grad():  # 训练集中不需要反向传播
    for data in test_loader:
        images, labels = data
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


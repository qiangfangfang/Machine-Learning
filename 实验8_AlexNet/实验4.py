# # 模型搭建
import torch.nn as nn
import torch
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

# import torch.nn as nn
# import torch
#
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(96, 256, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(256, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.classifier(x)
#         return x


import os 
import sys 
import json
import torch 
import torch.nn as nn 
from torchvision import transforms, datasets 
import torch.optim as optim 
from tqdm import tqdm #进度条打印所需包 代码进度可视化pip install tqdm 

# GPU or CPU选择 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), #随 机剪裁
                                 transforms.RandomHorizontalFlip(), #水平翻转
                                 transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))]), #标准化处理，均值方差参数为官 方参数
      "val": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}


BATCH_SIZE = 64 

path_train = r"F:\Class_Code\Pytorch_code\data_set\flower_data\train"
#数据路径 根据实际路径修改 
train_dataset = datasets.ImageFolder(root=path_train ,transform=data_transform["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=0)


# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} 
flower_list = train_dataset.class_to_idx 
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file 
json_str = json.dumps(cla_dict, indent=4)
# 编码 indent=4 代表5类 
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

path_val =r"F:\Class_Code\Pytorch_code\data_set\flower_data\val"
val_dataset = datasets.ImageFolder(root = path_val,transform=data_transform["val"])
val_num = len(val_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True, 
                                         num_workers=0)

net = AlexNet(num_classes=5).to(device) #导入模型 
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 
#优化器 学习率、动量参数 
loss_function = nn.CrossEntropyLoss() #损失函数 
epochs = 10 #迭代次数 
save_path = './AlexNet—W.pth' #每完成一次训练保存学习所得参数
best_acc = 0.0 
train_steps = len(train_loader)

print('Start Training') 
for epoch in range(epochs):
    # train 
    net.train() #启用dropout方法 
    running_loss = 0.0 # 统计训练过程中的平均损失 
    train_bar = tqdm(train_loader, file=sys.stdout) #打印训 练进度
    for step, data in enumerate(train_bar): 
    # 循环遍历训练集样本，通过enumerate函数能返回数据data和步数 step 
        images, labels = data 
        optimizer.zero_grad() # 历史损失梯度清零 
        logits = net(images.to(device)) 
        loss = loss_function(logits, labels.to(device)) 
        loss.backward() # loss反向传播 
        optimizer.step() # 优化器进行参数更新 
        
        # print statistics 
        running_loss += loss.item() #累加loss 
        
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs, loss)
        
    # validate 

    net.eval() 
    acc = 0.0 # accumulate accurate number / epoch 
    with torch.no_grad(): #无需计算误差梯度 
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data 
            outputs = net(val_images.to(device)) 
            # loss = loss_function(outputs, test_labels) 
            predict_y = torch.max(outputs, dim=1)[1] 
            # 最可能类别的预测 找到最大的index；
            # dim=1表示按行，取出每一行的最大值 ；
            # [0]:对应最大值；[1]对应最大值所在位置，index 
            acc += torch.eq(predict_y,val_labels.to(device)).sum().item()
            # 判断预测标签与真实标签是否相等，累积判断正确次数 
            
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

    val_accurate = acc / val_num #准确率计算 
    print('[epoch %d] train_loss: %.3f val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate)) 
    
    if val_accurate > best_acc: # 筛选出验证集上正确率最好的 一组模型参数
        best_acc = val_accurate 
        torch.save(net.state_dict(), save_path) 
        
print('Finished Training')
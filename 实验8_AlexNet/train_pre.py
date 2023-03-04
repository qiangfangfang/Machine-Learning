import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets

import torch.optim as optim
from tqdm import tqdm

from model_pre import AlexNet_Pre

# GPU or CPU选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

# 加载数据
BATCH_SIZE = 64

path_train = r"F:\Class_Code\Pytorch_code\data_set\flower_data\train"
train_dataset = datasets.ImageFolder(root=path_train , transform=data_transform["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=0)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)          # 编码
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

path_val = r"F:\Class_Code\Pytorch_code\data_set\flower_data\val"
val_dataset = datasets.ImageFolder(root = path_val, transform=data_transform["val"])
val_num = len(val_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=0)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))



net = AlexNet_Pre(num_classes=5).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)    #随机梯度下降
loss_function = nn.CrossEntropyLoss()

epochs = 10
save_path = './AlexNet_Pre.pth'
best_acc = 0.0
train_steps = len(train_loader)


print('Start Training')
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')


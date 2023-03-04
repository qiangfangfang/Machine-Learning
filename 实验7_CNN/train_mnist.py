import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16,
                                           shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16,
                                           shuffle=False, num_workers=0)

# import matplotlib.pyplot as plt
# figure = plt.figure()
# num_of_image = 60
#
# for imgs, targets in train_loader:
#     break
#
# for index in range (num_of_image):
#     plt.subplot(6,10,index+1)
#     plt.axis('off')
#     img=imgs[index,...]
#     plt.imshow(img.numpy().squeeze(),cmap='gray_r')
# plt.show()

from model import LeNet
import torch.nn as nn
import torch.optim as optim
model = LeNet().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs=10
total_step = len(train_loader)

for epoch in range(epochs):
    train_loss = 0.0
    for data, target in train_loader:
        data,target=data.to(device),target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))


correct = 0
total = 0
with torch.no_grad():  # 测试集中不需要反向传播
    for data in test_loader:
        images, labels = data
        images, labels =images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
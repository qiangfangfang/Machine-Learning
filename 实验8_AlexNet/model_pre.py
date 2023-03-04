import torch.nn as nn
import torch
from torchvision import models

class AlexNet_Pre(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_Pre, self).__init__()
        net = models.alexnet(pretrained=False)
        model_weight_path = "./alexnet-pre.pth"
        net.load_state_dict(torch.load(model_weight_path))
        net.classifier = nn.Sequential()  # classifier置空，重构全连接层
        self.features = net  # 保留特征提取网络
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

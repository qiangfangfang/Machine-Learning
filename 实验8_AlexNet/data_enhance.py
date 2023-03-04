import torch.nn as nn
import torch

import os
import torchvision.transforms as transforms
from PIL import Image

# 数据路径 根据实际情况修改
sourth_path_daisy = r'F:\Class_Code\Pytorch_code\data_set\flower_data\source\daisy'
sourth_path_roses = r'F:\Class_Code\Pytorch_code\data_set\flower_data\source\roses'
sourth_path_dandelion = r'F:\Class_Code\Pytorch_code\data_set\flower_data\source\dandelion'
sourth_path_sunflowers=r'F:\Class_Code\Pytorch_code\data_set\flower_data\source\sunflowers'
aim_dir_daisy = r'F:\Class_Code\Pytorch_code\data_set\flower_data\Enhance\daisy'
aim_dir_roses = r'F:\Class_Code\Pytorch_code\data_set\flower_data\Enhance\roses'
aim_dir_dandelion = r'F:\Class_Code\Pytorch_code\data_set\flower_data\Enhance\dandelion'
aim_dir_sunflowers=r'F:\Class_Code\Pytorch_code\data_set\flower_data\Enhance\sunflowers'
def dataEnhance(sourth_path,aim_dir,size):
    h = 0
    #得到目标文件的文件和文件名
    file_list = os.listdir(sourth_path)
    #创建目标文件夹
    if not os.path.exists(aim_dir):
        os.mkdir(aim_dir)
    #对目标文件夹内的文件进行遍历
    for i in file_list:
        img = Image.open('%s\%s'%(sourth_path, i))
        print(img.size)
        # Resize
        h = h + 1
        transform1 = transforms.Compose([transforms.ToTensor(),
                                         transforms.ToPILImage(),
                                         transforms.Resize(size)])
        img1 = transform1(img)
        img1.save('%s/%s.png'%(aim_dir,h))
        #颜色变换
        h = h + 1
        transform2 = transforms.Compose([transforms.ToTensor(),
                                         transforms.ToPILImage(),
                                         transforms.ColorJitter(
                                             brightness=0.5, contrast=0.5,
                                             saturation=0.5, hue=0.5),
                                         transforms.Resize(size)])
        img2 = transform2(img)
        img2.save('%s/%s.png'%(aim_dir,h))
        # 随机剪裁
        h = h + 1
        transform3 = transforms.Compose([transforms.ToTensor(),
                                         transforms.ToPILImage(),
                                         transforms.RandomResizedCrop(size)])
        img3 = transform3(img)
        img3.save('%s/%s.png'%(aim_dir,h))

        # 旋转变换
        h = h + 1
        transform4 = transforms.Compose([transforms.ToTensor(),
                                         transforms.ToPILImage(),
                                         transforms.RandomRotation(60),  #旋转角度
                                         transforms.Resize(size)])
        img4 = transform4(img)
        img4.save('%s/%s.png'%(aim_dir,h))

dataEnhance(sourth_path_daisy,aim_dir_daisy,(224,224))
dataEnhance(sourth_path_roses,aim_dir_roses,(224,224))
dataEnhance(sourth_path_dandelion,aim_dir_dandelion,(224,224))
dataEnhance(sourth_path_sunflowers,aim_dir_sunflowers,(224,224))
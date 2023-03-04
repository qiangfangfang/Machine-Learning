import os
import torchvision.transforms as transforms
from PIL import Image

sourth_path_daisy = r'F:\Class_Code\Pytorch_code\data_set\flower_data\source\daisy'
aim_dir_daisy = r'F:\Class_Code\Pytorch_code\data_set\flower_data\enhance'

def dataEnhance(sourth_path, aim_dir, size):
    h = 0
    file_list = os.listdir(sourth_path)  # 得到目标文件的文件和 文件名
    if not os.path.exists(aim_dir):  # 创建目标文件夹
        os.mkdir(aim_dir)
    for i in file_list:  # 对目标文件夹内的文件 进行遍历
        img = Image.open('%s\%s' % (sourth_path, i))
        print(img.size)

        # (1) Resize
        h = h + 1
        transform1 = transforms.Compose([transforms.ToTensor(),

                                         transforms.ToPILImage(),

                                         transforms.Resize(size)])
        img1 = transform1(img)
        img1.save('%s/%s.png' % (aim_dir, h))

        # (2) 颜色变换
        h = h + 1
        transform2 = transforms.Compose([transforms.ToTensor(),

                                         transforms.ToPILImage(),

                                         transforms.ColorJitter(

                                             brightness=0.5, contrast=0.5,

                                             saturation=0.5, hue=0.5),

                                         transforms.Resize(
                                             size)])  # brightness(亮度)、contrast(对比度)、saturation(饱和 度)、hue(色调)
        img2 = transform2(img)
        img2.save('%s/%s.png' % (aim_dir, h))

        # (3) 随机剪裁
        h = h + 1
        transform3 = transforms.Compose([transforms.ToTensor(),
                                         transforms.ToPILImage(),
                                         transforms.RandomResizedCrop(size)])
        img3 = transform3(img)
        img3.save('%s/%s.png' % (aim_dir, h))

        # (4) 旋转变换
        h = h + 1
        transform4 = transforms.Compose([transforms.ToTensor(),
                                         transforms.ToPILImage(),
                                         transforms.RandomRotation(60),  # 旋转角度

                                         transforms.Resize(size)])
        img4 = transform4(img)
        img4.save('%s/%s.png' % (aim_dir, h))


dataEnhance(sourth_path_daisy, aim_dir_daisy, (224, 224))


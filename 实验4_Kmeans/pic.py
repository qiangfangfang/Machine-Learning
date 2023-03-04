from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("Lenna.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
# plt.imshow(image)
# # plt.show()

# 像素点归一化、去重
import pandas as pd
# 归一化
image_norm = image / image.max()
# 保留3通道
image_arr = image_norm.reshape(-1, 3)
# print(image_arr) #(273280, 3)
# 像素点去重
px_uni = pd.DataFrame(image_arr).drop_duplicates()
# print(px_uni.shape) #(96615, 3)

# KMeans找质心
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

# 定义聚类簇的数量
n_clusters = 64

# 随机取前2000个点
px_sample = shuffle(px_uni, random_state=42)[:2000]
# 以2000个点进行聚类找质心
cluster = KMeans(n_clusters, random_state=0)
cluster.fit(px_sample)
# 质心
centers = cluster.cluster_centers_

# 用质心替换图片像素点
import numpy as np

labels = cluster.predict(image_arr)
# 按索引找到对应质心
image_cg = centers[labels]
# 图片像素还原
image_cg = (image_cg * image.max()).reshape(image.shape).astype('uint8')

#图片对比
plt.figure(1)
plt.imshow(image)
plt.axis('off')

plt.figure(2)
plt.imshow(image_cg)
plt.axis('off')
plt.show()

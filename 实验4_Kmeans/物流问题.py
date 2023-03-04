from numpy import *
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import numpy as np

def distEclud(vecA, vecB):
   return sqrt(sum(power(vecA-vecB, 2)))  # 欧氏距离

def nearest(data, cluster_centers, distMeas=distEclud):
   min_dist = np.inf
   m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
   for i in range(m):
      d = distMeas(data, cluster_centers[i,])  # 计算point与每个聚类中心之间的距离
      if d < min_dist:  # 选择最短距离
         min_dist = d
   return min_dist


def get_centroids(data, k, distMeas=distEclud):
   print('2.initalize cluster center')
   m, n = np.shape(data)
   cluster_centers = np.mat(np.zeros((k, n)))
   index = np.random.randint(0, m)  # 1、随机选择一个样本点为第一个聚类中心
   cluster_centers[0,] = np.copy(data[index,])
   d = [0.0 for _ in range(m)]  # 2、初始化一个距离的序列
   for i in range(1, k):
      sum_all = 0
      for j in range(m):
         d[j] = nearest(data[j,], cluster_centers[0:i, ], distMeas)  # 3、对每一个样本找到最近的聚类中心点
         sum_all += d[j]  # 4、将所有的最短距离相加
      sum_all *= random.random()  # 5、取得sum_all之间的随机值
      for j, di in enumerate(d):  # 6、获得距离最远的样本点作为聚类中心点
         sum_all -= di
         if sum_all > 0:
            continue
         cluster_centers[i] = np.copy(data[j,])
         break
   return cluster_centers

def kMeans(dataSet, k, distMeas=distEclud, createCent=get_centroids):
   m = shape(dataSet)[0]               # 数据数目
   clusterAssment = mat(zeros((m, 2))) # 各簇中的数据点
   centroids = createCent(dataSet, k)  # 各簇质心生成
   clusterChanged = True
   print('3.recompute and reallocated...')
   while clusterChanged:       #重复计算，直到簇分配不再变化
      clusterChanged = False
      # 1.每个数据分配到最近的簇中
      for i in range(m):
         minDist = inf
         minIndex = -1
         for j in range(k):
            distJI = distMeas(centroids[j, :], dataSet[i, :])
            if distJI < minDist:
               minDist = distJI
               minIndex = j
         if clusterAssment[i, 0] != minIndex:  # 如果之前记录的簇索引不等于目前最小距离的簇索引
            clusterChanged = True  # 设置为True，继续遍历，直到簇分配结果不再改变为止
         clusterAssment[i, :] = minIndex, minDist**2  # 记录新的簇索引和误差
      # 2.更新质心的位置
      for cent in range(k):
         ptsInclust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取给定簇的所有点
         centroids[cent, :] = mean(ptsInclust, axis=0)  # 然后计算均值，axis=0沿着列方向
   return centroids, clusterAssment  # 返回质心与点分配结果


def show(dataset,k,classCenter,clusterPoints):
    print('4.load the map')
    fig=plt.figure()
    rect=[0.1,0.1,1.0,1.0]
    axprops=dict(xticks=[],yticks=[])
    # ax0=fig.add_axes(rect,label='ax0',**axprops) #frameon=False)
    ax0 = fig.add_axes(rect, label='ax1', frameon=False)
    imgp=plt.imread('city.png')
    ax0.imshow(imgp)
    ax1=fig.add_axes(rect,label='ax1',frameon=False)
    print('5.show the clusters')
    numsamples=len(dataset)
    mark=['ok','^b','om','og','sc']
    for i in range(numsamples):
        markindex=int(clusterPoints[i,0])%k
        ax1.plot(dataset[i,0],dataset[i,1],mark[markindex])
    for i in range(k):
        markindex=int(clusterPoints[i,0])%k
        ax1.plot(classCenter[i,0],classCenter[i,1],'^r',markersize=12)
    plt.show()
print('1. load the dataset')
dataset=loadtxt('testSet_logistics.txt')
k=5
classCenter,clssspoints=kMeans(dataset,k)
show(dataset,k,classCenter,clssspoints)

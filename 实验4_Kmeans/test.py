from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def loadDateSet(fileName):
   dataMat = []
   fr = open(fileName)  # 打开文件
   for line in fr.readlines():  # 遍历文件的每一行(每行表示一个数据)
      curLine = line.strip().split('\t')  # 处理每行数据，返回字符串list
      fltLine = list(map(float, curLine))  # 使用float函数处理list中的字符串，使其为float类型
      dataMat.append(fltLine)  # 将该数据加入到数组中
   return dataMat

def distEclud(vecA, vecB):
   return sqrt(sum(power(vecA-vecB, 2)))  # 欧氏距离

def randCent(dataSet, k):
   print('initialize cluster center....')
   n = shape(dataSet)[1]  # 数据特征个数(即数据维度)
   # 创建一个0矩阵，其中zeros为创建0填充的数组，mat是转换为矩阵，用于存放k个质心
   centroids = mat(zeros((k, n)))
   for i in range(n):  # 遍历每个特征
      minI = min(dataSet[:,i])  # 获取最小值
      rangeI = float(max(dataSet[:, i]) - minI)  # 范围
      centroids[:, i] = minI + rangeI * random.rand(k, 1)  # 最小值+范围*随机数
   return centroids

datMat = mat(loadDateSet('testSet.txt'))
print(randCent(datMat, 2))

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
   m = shape(dataSet)[0]               # 数据数目
   clusterAssment = mat(zeros((m, 2))) # 各簇中的数据点
   centroids = createCent(dataSet, k)  # 各簇质心生成
   clusterChanged = True
   print('recompute and reallocated...')
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

datMat = mat(loadDateSet('testSet.txt'))
myCentroids, clusterAssing = kMeans(datMat, 4)

marker = ['s', 'o', '^', '<']  # 散点图点的形状
color = ['b','m','c','g']  # 颜色

X = np.array(datMat)  # 数据点
CentX = np.array(myCentroids)  # 质心点4个
Cents = np.array(clusterAssing[:,0])  # 每个数据点对应的簇
for i,Centroid in enumerate(Cents):  # 遍历每个数据对应的簇，返回数据的索引即其对应的簇
   plt.scatter(X[i][0], X[i][1], marker=marker[int(Centroid[0])],c=color[int(Centroid[0])])  # 按簇画数据点
plt.scatter(CentX[:,0],CentX[:,1],marker='*',c = 'r')  # 画4个质心
plt.show()
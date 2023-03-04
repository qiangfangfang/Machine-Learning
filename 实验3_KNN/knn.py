# -*- coding: utf-8 -*-

import numpy as np
import operator


# k-近邻算法
def knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    diffMat = np.array(inX) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

# 训练数据
def createDataSet():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
        [0.2,0.2]
    ])
    labels = ['A', 'A', 'B', 'B','B']
    return group, labels

# 查看数据
group, labels = createDataSet()
print("训练数据:", group)
print("标签:", labels)

# 预测
result = knn([1.2, 1], group, labels, 3)
print("预测标签为：", result)

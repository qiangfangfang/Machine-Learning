# -*- coding: utf-8 -*-

import numpy as np
import operator
import pandas as pd


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

# path = 'data/iris.data'
# data = pd.read_csv(path, header=None)
# data=data.values
# x_sample = data[:,:-1]
# label = data[:,-1]
from sklearn import datasets
iris = datasets.load_iris()
x_sample=iris.data
label=iris.target
print(x_sample)
print(label)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_sample, label, test_size=0.2, random_state=42)





# 预测
# X_test[0]=[5.1,3.5,1.4,0.2] #改错一个数据
for k in range(1,11):
    pre = 0
    for i in range(len(X_test)):
        res = knn(X_test[i], X_train, y_train, k)
        if res == y_test[i]:
            pre = pre + 1
    acc = pre/len(X_test)
    print("k=",k,"正确率为：",acc)

# 预测
# inx= [6.7,3.0,5.2,2.3]
# result = knn(inx, x_sample, label, 3)
# print("预测标签为：", result)






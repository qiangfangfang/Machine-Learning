# -*- coding: utf-8 -*-

import numpy as np
import operator
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

def knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
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

# 将图像格式转化为向量 32*32 --> 1*1024
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# 测试代码
# testVector = img2vector('datas/testDigits/0_13.txt')
# print(testVector[0, 0:31])
# print(testVector[0, 32:63])


# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('./data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./data/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('./data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./data/testDigits/%s' % fileNameStr)
        classifierResult = knn(vectorUnderTest, trainingMat, hwLabels, 5)
        print("预测结果为：%d，真实值为：%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("预测错误的总数为：%d" % errorCount)
    print("手写数字识别系统的错误率为：%f" % (errorCount / float(mTest)))

# 测试手写数字识别系统
handwritingClassTest()
# encoding=utf-8
import numpy as np
import DBSCAN as db
import kmeans as km


def calcDistByMin(dataMat, ck1, ck2):  # 最小距离点作为簇间的距离
    min = np.inf
    for vec1 in ck1:
        for vec2 in ck2:
            dist = km.getDistance(dataMat[vec1, :], dataMat[vec2, :])
            if dist <= min:
                min = dist
    return min


def calcDistByMax(dataMat, ck1, ck2):  # 最大距离点作为簇间的距离
    max = 0
    for vec1 in ck1:
        for vec2 in ck2:
            dist = km.getDistance(dataMat[vec1, :], dataMat[vec2, :])
            if dist >= max:
                max = dist
    return max


def createDistMat(dataMat, calcDistType=calcDistByMin):  # 生成初始的距离矩阵
    m = dataMat.shape[0]
    distMat = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(m):
            listI = [i];
            listJ = [j]  # 为配合距离函数的输入参数形式，在这里要列表化一下
            distMat[i, j] = calcDistType(dataMat, listI, listJ)
            distMat[j, i] = distMat[i, j]
    return distMat


def findMaxLoc(distMat, q):  # 寻找矩阵中最小的元素并返回其位置，注意，这里不能返回相同的坐标
    min = np.inf
    I = J = 0
    for i in range(q):
        for j in range(q):
            if distMat[i, j] < min and i != j:
                min = distMat[i, j]
                I = i
                J = j
    return I, J


def ANGES(dataMat, k, calcDistType=calcDistByMax):
    m = dataMat.shape[0]
    ck = []
    for i in range(m):
        ck.append([i])
    distMat = createDistMat(dataMat, calcDistType)
    q = m  # 初始化点集个数
    while q > k:
        i, j = findMaxLoc(distMat, q)
        # print i,j
        if i > j:
            i, j = j, i  # 保证i<j，这样做是为了删除的是序号较大的簇
        ck[i].extend(ck[j])  # 把序号较大的簇并入序号小的簇
        del ck[j]  # 删除序号大的簇
        distMat = np.delete(distMat, j, 0)  # 在距离矩阵中删除该簇的数据，注意这里delete函数有返回值，否则不会有删除作用
        distMat = np.delete(distMat, j, 1)
        print
        distMat.shape
        for index in range(0, q - 1):  # 重新计算新簇和其余簇之间的距离
            distMat[i, index] = calcDistType(dataMat, ck[i], ck[index])
            distMat[i, index] = distMat[index, i]
        q -= 1  # 一个点被分入簇中，自减
    return ck


if __name__ == '__main__':
    dataMat = km.loadDataSet("testSet.txt")
    ck = ANGES(dataMat, 4)
    print(ck)
    db.plotAns(dataMat, ck)

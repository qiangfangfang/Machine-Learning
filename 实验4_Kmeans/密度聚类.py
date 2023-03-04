# encoding=utf-8
import numpy as np
import kmeans as km
import matplotlib.pyplot as plt


def createDisMat(dataMat):
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    distMat = np.mat(np.zeros((m, m)))  # 初始化距离矩阵，这里默认使用欧式距离
    for i in range(m):
        for j in range(m):
            if i == j:
                distMat[i, j] = 0
            else:
                dist = km.getDistance(dataMat[i, :], dataMat[j, :])
                distMat[i, j] = dist
                distMat[j, i] = dist
    return distMat


def findCore(dataMat, delta, minPts):
    core = []
    m = dataMat.shape[0]
    n = dataMat.shape[1]
    distMat = createDisMat(dataMat)
    for i in range(m):
        temp = distMat[i, :] < delta  # 单独抽取矩阵一行做过滤，凡是小于邻域值的都被标记位True类型
        ptsNum = np.sum(temp, 1)  # 按行加和，统计小于邻域值的点个数
        if ptsNum >= minPts:
            core.append(i)  # 满足条件，增加核心点
    return core


def DBSCAN(dataMat, delta, minPts):
    k = 0
    m = dataMat.shape[0]
    distMat = createDisMat(dataMat)  # 获取距离矩阵
    core = findCore(dataMat, delta, minPts)  # 获取核心点列表
    unVisit = [1] * m  # hash值作为标记，当某一位置的数据位1时，表示还未被访问，为0表示已经被访问
    Q = []
    ck = []
    unVistitOld = []
    while len(core) != 0:
        print('a')
        unVistitOld = unVisit[:]  # 保留原始的未被访问集
        i = np.random.choice(core)  # 在核心点集中随机选择样本
        Q.append(i)  # 加入对列Q
        unVisit[i] = 0  # 剔除当前加入对列的数据，表示已经访问到了
        while len(Q) != 0:
            print
            len(Q)
            temp = distMat[Q[0], :] < delta  # 获取在此核心点邻域范围内的点集
            del Q[0]
            ptsNum = np.sum(temp, 1)
            if ptsNum >= minPts:
                for j in range(len(unVisit)):
                    if unVisit[j] == 1 and temp[0, j] == True:
                        Q.append(j)
                        unVisit[j] = 0
        k += 1
        ck.append([])
        for index in range(m):
            if unVistitOld[index] == 1 and unVisit[index] == 0:  # 上一轮未被访问到此轮被访问到的点均要加入当前簇
                ck[k - 1].append(index)
                if index in core:  # 在核心点集中清除当前簇的点
                    del core[core.index(index)]
    return ck


def plotAns(dataSet, ck):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[ck[0], 0], dataSet[ck[0], 1], c='blue')
    ax.scatter(dataSet[ck[1], 0], dataSet[ck[1], 1], c='red')
    ax.scatter(dataSet[ck[2], 0], dataSet[ck[2], 1], c='green')
    ax.scatter(dataSet[ck[3], 0], dataSet[ck[3], 1], c='yellow')

    # ax.scatter(centRoids[:,0],centRoids[:,1],c = 'red',marker = '+',s = 70)
    plt.show()


if __name__ == '__main__':
    dataMat = km.loadDataSet("testSet.txt")
    # distMat = createDisMat(dataMat)
    # core = findCore(dataMat,1,5)
    # print distMat
    # print len(core)
    ck = DBSCAN(dataMat, 2, 15)
    print(ck)
    print(len(ck))
    plotAns(dataMat, ck)


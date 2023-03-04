###建立kd树和实现查询功能
import numpy as np
import matplotlib.pyplot as plt
import time


class kdTree:
    def __init__(self, parent_node):
        '''
        节点初始化
        '''
        self.nodedata = None  ###当前节点的数据值，二维数据
        self.split = None  ###分割平面的方向轴序号,0代表沿着x轴分割，1代表沿着y轴分割
        self.range = None  ###分割临界值
        self.left = None  ###左子树节点
        self.right = None  ###右子树节点
        self.parent = parent_node  ###父节点
        self.leftdata = None  ###保留左边节点的所有数据
        self.rightdata = None  ###保留右边节点的所有数据
        self.isinvted = False  ###记录当前节点是否被访问过

    def print(self):
        '''
        打印当前节点信息
        '''
        print(self.nodedata, self.split, self.range)

    def getSplitAxis(self, all_data):
        '''
        根据方差决定分割轴
        '''
        var_all_data = np.var(all_data, axis=0)
        if var_all_data[0] > var_all_data[1]:
            return 0
        else:
            return 1

    def getRange(self, split_axis, all_data):
        '''
        获取对应分割轴上的中位数据值大小
        '''
        split_all_data = all_data[:, split_axis]
        data_count = split_all_data.shape[0]
        med_index = int(data_count / 2)
        sort_split_all_data = np.sort(split_all_data)
        range_data = sort_split_all_data[med_index]
        return range_data

    def getNodeLeftRigthData(self, all_data):
        '''
        将数据划分到左子树，右子树以及得到当前节点
        '''
        data_count = all_data.shape[0]
        ls_leftdata = []
        ls_rightdata = []
        for i in range(data_count):
            now_data = all_data[i]
            if now_data[self.split] < self.range:
                ls_leftdata.append(now_data)
            elif now_data[self.split] == self.range and self.nodedata == None:
                self.nodedata = now_data
            else:
                ls_rightdata.append(now_data)
        self.leftdata = np.array(ls_leftdata)
        self.rightdata = np.array(ls_rightdata)

    def createNextNode(self, all_data):
        '''
        迭代创建节点，生成kd树
        '''
        if all_data.shape[0] == 0:
            print("create kd tree finished!")
            return None
        self.split = self.getSplitAxis(all_data)
        self.range = self.getRange(self.split, all_data)
        self.getNodeLeftRigthData(all_data)
        if self.leftdata.shape[0] != 0:
            self.left = kdTree(self)
            self.left.createNextNode(self.leftdata)
        if self.rightdata.shape[0] != 0:
            self.right = kdTree(self)
            self.right.createNextNode(self.rightdata)

    def plotKdTree(self):
        '''
        在图上画出来树形结构的递归迭代过程
        '''
        if self.parent == None:
            plt.figure(dpi=300)
            plt.xlim([0.0, 10.0])
            plt.ylim([0.0, 10.0])
        color = np.random.random(3)
        if self.left != None:
            plt.plot([self.nodedata[0], self.left.nodedata[0]], [self.nodedata[1], self.left.nodedata[1]], '-o',
                     color=color)
            plt.arrow(x=self.nodedata[0], y=self.nodedata[1], dx=(self.left.nodedata[0] - self.nodedata[0]) / 2.0,
                      dy=(self.left.nodedata[1] - self.nodedata[1]) / 2.0, color=color, head_width=0.2)
            self.left.plotKdTree()
        if self.right != None:
            plt.plot([self.nodedata[0], self.right.nodedata[0]], [self.nodedata[1], self.right.nodedata[1]], '-o',
                     color=color)
            plt.arrow(x=self.nodedata[0], y=self.nodedata[1], dx=(self.right.nodedata[0] - self.nodedata[0]) / 2.0,
                      dy=(self.right.nodedata[1] - self.nodedata[1]) / 2.0, color=color, head_width=0.2)
            self.right.plotKdTree()
        # if self.split == 0:
        #     x = self.range
        #     plt.vlines(x, 0, 10, color=color, linestyles='--')
        # else:
        #     y = self.range
        #     plt.hlines(y, 0, 10, color=color, linestyles='--')



# test_array = 10.0 * np.random.random([30, 2])
test_array = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
my_kd_tree = kdTree(None)
my_kd_tree.createNextNode(test_array)
my_kd_tree.plotKdTree()
plt.show()




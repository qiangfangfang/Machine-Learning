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

    # kd树上的最近邻查找算法
    def divDataToLeftOrRight(self, find_data):
        '''
        根据传入的数据将其分给左节点(0)或右节点(1)
        '''
        data_value = find_data[self.split]
        if data_value < self.range:
            return 0
        else:
            return 1

    def getSearchPath(self, ls_path, find_data):
        '''
        二叉查找到叶节点上
        '''
        now_node = ls_path[-1]
        if now_node == None:
            return ls_path
        now_split = now_node.divDataToLeftOrRight(find_data)
        if now_split == 0:
            next_node = now_node.left
        else:
            next_node = now_node.right
        while (next_node != None):
            ls_path.append(next_node)
            next_split = next_node.divDataToLeftOrRight(find_data)
            if next_split == 0:
                next_node = next_node.left
            else:
                next_node = next_node.right
        return ls_path

    def getNestNode(self, find_data, min_dist, min_data):
        '''
        回溯查找目标点的最近邻距离
        '''
        ls_path = []
        ls_path.append(self)
        self.getSearchPath(ls_path, find_data)
        now_node = ls_path.pop()
        now_node.isinvted = True
        min_data = now_node.nodedata
        min_dist = np.linalg.norm(find_data - min_data)
        while (len(ls_path) != 0):
            back_node = ls_path.pop()  ### 向上回溯一个节点
            if back_node.isinvted == True:
                continue
            else:
                back_node.isinvted = True
            back_dist = np.linalg.norm(find_data - back_node.nodedata)
            if back_dist < min_dist:
                min_data = back_node.nodedata
                min_dist = back_dist
            if np.abs(find_data[back_node.split] - back_node.range) < min_dist:
                ls_path.append(back_node)
                if back_node.left.isinvted == True:
                    if back_node.right == None:
                        continue
                    ls_path.append(back_node.right)
                else:
                    if back_node.left == None:
                        continue
                    ls_path.append(back_node.left)
                ls_path = back_node.getSearchPath(ls_path, find_data)
                now_node = ls_path.pop()
                now_node.isinvted = True
                now_dist = np.linalg.norm(find_data - now_node.nodedata)
                if now_dist < min_dist:
                    min_data = now_node.nodedata
                    min_dist = now_dist
        print("min distance:{}  min data:{}".format(min_dist, min_data))
        return min_dist, min_data

    def getNestDistByEx(self, test_array, find_data, min_dist, min_data):
        '''
        穷举法得到目标点的最近邻距离
        '''
        data_count = test_array.shape[0]
        min_data = test_array[0]
        min_dist = np.linalg.norm(find_data - min_data)
        for i in range(data_count):
            now_data = test_array[i]
            now_dist = np.linalg.norm(find_data - now_data)
            if now_dist < min_dist:
                min_dist = now_dist
                min_data = now_data
        print("min distance:{}  min data:{}".format(min_dist, min_data))
        return min_dist,min_data



# test_array = 10.0*np.random.random([50,2])   ## 验证算法
# test_array = 10.0*np.random.random([100000,2])### 随机生成n个2维0-10以内的数据点
test_array =np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
my_kd_tree = kdTree(None)                    ### kd树实例化
my_kd_tree.createNextNode(test_array)        ### 生成kd树
# my_kd_tree.plotKdTree()
find_data = np.array([3, 4.5])             ### 待查找目标点
min_dist = 0                                 ### 临时变量，存储最短距离
min_data = np.array([0.0, 0.0])              ### 临时变量，存储取到最短距离时对应的数据点

t1 = time.perf_counter()
min_dist,min_data = my_kd_tree.getNestNode(find_data, min_dist, min_data)        ### 利用kd树回溯查找
t_1 = time.perf_counter()-t1
print(t_1)
#
# t2=time.perf_counter()
# min_dist,min_data = my_kd_tree.getNestDistByEx(test_array, find_data, min_dist, min_data)    ### 穷举法查找
# t_2 = time.perf_counter()-t2
# print(t_2)


# plt.figure(dpi=300)
# plt.xlim([0.0, 10.0])
# plt.ylim([0.0, 10.0])
# plt.scatter(test_array[:,0],test_array[:,1])
# plt.plot([5.0],[5.0],'r*') #marker='*'
# plt.plot(min_data[0],min_data[1],'y*')
# plt.show()


from math import log
# 计算给定数据集的香农熵
# H(x) = -sum{p(i)log[p(i)]}

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)  # 计算数据集的数目
	labelCounts = {}  # 创建空字典，key为标签，value为数据集中为key标签的数据总数目
	for featVec in dataSet:  # 遍历每条数据
		currentLabel = featVec[-1]  # 获取当前数据的标签
		if currentLabel not in labelCounts.keys():  # 判断当前标签是否在字典中，不在字典中就扩展字典，并将value设为0
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1  # 如果字典中已经存在此标签，就将其value加1，即数目加1
	shannonEnt = 0.0  # 香农熵
	for key in labelCounts:  # 遍历字典，计算每个标签各占总数据集的比例
		prob = float(labelCounts[key]) / numEntries  # 获取该标签的value，除以数据集的数目
		shannonEnt -= prob * log(prob, 2)  # 按香农熵公式计算香农熵

	return shannonEnt

def createDataSet():
    dataSet = [
        ['青绿','蜷缩','浊响','清晰','凹陷','硬滑', 'yes'],
        ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑', 'yes'],
        ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑', 'yes'],
        ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑', 'yes'],
        ['浅白','蜷缩','浊响','清晰','凹陷','硬滑', 'yes'],
		['青绿','稍蜷','浊响','清晰','稍凹','软粘', 'yes'],
		['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 'yes'],
		['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 'yes'],
		['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 'no'],
		['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 'no'],
		['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 'no'],
		['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 'no'],
		['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 'no'],
		['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 'no'],
		['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 'no'],
		['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 'no'],
		['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 'no'],
    ]
    labels = ['色泽', '根蒂','敲声','纹理','脐部','触感']
    return dataSet, labels

#测试
Dataset,labels = createDataSet()
Ent = calcShannonEnt(Dataset)
print("熵：",Ent)


# dataSet中找到axis=value的样本
# 将这些样本删去第axis个标签值
# 最后得到比原数据集少了一列属性并且只有删除的那列属性值为value的数据集
def splitDataSet(dataSet, axis, value):
	#dataSet：待划分的数据集、axis:划分数据集特征、value:需要返回的特征的值
    sub_DataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            sub_DataSet.append(reducedFeatVec)
    return sub_DataSet

# # 测试划分数据集
Dataset,labels =createDataSet()
sub_DataSet=splitDataSet(Dataset,0,'浅白')
print(len(sub_DataSet))
print(sub_DataSet)

# 选择最优划分特征
# Gain = H(D) - H(D|A)
def chooseBestFeatureToSplit(dataSet):
    # 计算特征属性的个数
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0  # 初始化信息增益
    bestFeature_index = -1  # 初始化最佳特征下标
    # 遍历每个属性
    for i in range(numFeatures):
        # 构造所有样本在当前特征的取值的列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 初始化信息熵：H(D|A)计算
        newEntropy = 0.0
        # 遍历该属性中的每个可能的属性值
        for value in uniqueVals:
            sub_DataSet = splitDataSet(dataSet, i, value)
            # 计算该属性值为value的占整个数据集的比例
            prob = len(sub_DataSet) / float(len(dataSet))
            # 计算使用该特征进行样本划分后的新信息熵
            newEntropy += prob * calcShannonEnt(sub_DataSet)
        infoGain = baseEntropy - newEntropy  # 求信息增益
        if infoGain > bestInfoGain:  # 如果此信息增益大于之前的记录的信息增益
            bestInfoGain = infoGain  # 更新最大的信息增益
            bestFeature_index = i
    return bestFeature_index,bestInfoGain

# # 测试 选择最优划分属性
Dataset,labels =createDataSet()
bestFeature_index,bestInfoGain=chooseBestFeatureToSplit(Dataset)
labels = ['色泽', '根蒂','敲声','纹理','脐部','触感']
print(labels[bestFeature_index])
print(bestInfoGain)


# 返回最多样本数的那个标签的值
def majority_label(classList):
	# 初始化统计各标签次数的字典
	# 键为各标签，对应的值为标签次数
    classCount = {}
    for key in classList:
        if key not in classCount.keys():
        	classCount[key] = 0
        classCount[key] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=lambda a:a[1], reverse=True)
    return sortedClassCount[0][0]  # 取出第一个，即投票最多的

# 递归生成决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 递归停止的第一个条件：当前集合所有样本标签相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归停止的第二个条件：遍历完所有特征，样本仍未被分“纯”
	# 使用majority_label，返回最多样本数的那个标签的值
    if len(dataSet[0]) == 1:
        return majority_label(classList)
	# 找出最佳数据集划分的特征属性的索引
    bestFeature_index,bestInfoGain = chooseBestFeatureToSplit(dataSet)
    best_Feature = labels[bestFeature_index]  # 获取最佳索引对应的值
	#初始化决策树
    Decision_tree = {best_Feature: {}}
	# 使用过当前最佳特征后将其删除
    del (labels[bestFeature_index])
	# 取出各样本在当前最佳特征上的取值列表
    featValues = [example[bestFeature_index] for example in dataSet]
	# 用set构造当前最佳特征取值的不重复集合
    uniqueVals = set(featValues)
	# 遍历所有最佳划分的分类标签
    for value in uniqueVals:
		# 子特征=当前特征（已经删去用过特征）
        subLabels = labels[:]
        Decision_tree[best_Feature][value] = createTree(splitDataSet(dataSet, bestFeature_index, value), subLabels)  # 递归调用
    return Decision_tree

# 测试
Dataset,labels =createDataSet()
Decision_tree = createTree(Dataset, labels)
print(Decision_tree)


# 用训练好的决策树对新样本分类
def classify(decision_tree, features, test_example):
    # 根节点代表的属性
	first_feature = list(decision_tree.keys())[0]
    # second_dict是第一个分类属性的值（也是字典）
	second_dict = decision_tree[first_feature]
    # 树根代表的属性，所在属性标签中的位置，即第几个属性
	index_of_first_feature = features.index(first_feature)
    # 对于second_dict中的每一个key
	for key in second_dict.keys():
		if test_example[index_of_first_feature] == key:
            # 若当前second_dict的key的value是一个字典
			if type(second_dict[key]).__name__ == 'dict':
                # 则需要递归查询
				classLabel = classify(second_dict[key], features, test_example)
            # 若当前second_dict的key的value是一个单独的值
			else:
                # 则就是要找的标签值
				classLabel = second_dict[key]
	return classLabel

# 新样本测试
labels = ['色泽', '根蒂','敲声','纹理','脐部','触感']
test_example = ['青绿', '稍蜷', '浊响', '模糊', '平坦', '软粘']
print('\n',classify(Decision_tree, labels, test_example))
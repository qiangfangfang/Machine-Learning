# 构造数据集
def createDataSet():
    dataset = [['youth', 'no', 'no', 'just so-so', 'no'],
               ['youth', 'no', 'no', 'good', 'no'],
               ['youth', 'yes', 'no', 'good', 'yes'],
               ['youth', 'yes', 'yes', 'just so-so', 'yes'],
               ['youth', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'good', 'no'],
               ['midlife', 'yes', 'yes', 'good', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'great', 'yes'],
               ['geriatric', 'no', 'no', 'just so-so', 'no']]
    features = ['age', 'work', 'house', 'credit']
    return dataset, features

# 计算当前集合的Gini系数
def calcGini(dataset):
    numEntries = len(dataset)
    labelCnt = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCnt.keys():
            labelCnt[currentLabel] = 0
        labelCnt[currentLabel] += 1
    # 得到了当前集合中每个标签的样本个数后，计算它们的p值
    prob_2 = 0
    for key in labelCnt:
        prob = float(labelCnt[key]) / numEntries
        prob_2 += prob * prob
    # 计算Gini系数
    Gini = 1 - prob_2
    return Gini
# 测试
Dataset,labels = createDataSet()
Gini = calcGini(Dataset)
print("基尼系数：",Gini)

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

# 将当前样本集分割成特征i取值为value的一部分和取值不为value的一部分（二分）
def create_sub_dataset(dataset, axis, value):
    sub_dataset1 = []
    sub_dataset2 = []
    for featVec in dataset:
        current_list = []
        if featVec[axis] == value:
            current_list = featVec[:axis]
            current_list.extend(featVec[axis + 1:])
            sub_dataset1.append(current_list)
        else:
            current_list = featVec[:axis]
            current_list.extend(featVec[axis + 1:])
            sub_dataset2.append(current_list)
    return sub_dataset1, sub_dataset2

# 测试
Dataset,labels =createDataSet()
sub_dataset1,sub_dataset2=create_sub_dataset(Dataset,0,'youth')
print('\n',sub_dataset1)
print('\n',sub_dataset2)

# 选取最优划分属性
def chooseBestFeatureToSplit(dataSet):
    # 计算特征属性的个数
    numFeatures = len(dataSet[0]) - 1
    if numFeatures == 1:  # 当只有一个特征时
        return 0
    bestGini = 1 # 初始化最佳基尼系数
    bestFeature_index = -1 # 初始化最佳特征下标
    # 遍历每个属性
    for i in range(numFeatures):
        # 构造所有样本在当前特征的取值的列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  #去重
        Gini = {}
        # 遍历该属性中的每个可能的属性值
        for value in uniqueVals:
            # 先求由该值进行划分得到的两个子集
            sub_dataset1, sub_dataset2 = create_sub_dataset(dataSet, i, value)
            # 求两个子集占原集合的比例系数prob1 prob2
            prob1 = len(sub_dataset1) / float(len(dataSet))
            prob2 = len(sub_dataset2) / float(len(dataSet))
            # 计算子集1的Gini系数
            Gini_of_sub_dataset1 = calcGini(sub_dataset1)
            # 计算子集2的Gini系数
            Gini_of_sub_dataset2 = calcGini(sub_dataset2)
            # 计算由当前最优切分点划分后的最终Gini系数
            Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2

        if Gini[value] < bestGini:
            bestGini = Gini[value]
            bestFeature_index = i
            best_Split_Point = value
    return bestFeature_index,best_Split_Point,bestGini

# 测试
Dataset,labels =createDataSet()
bestFeature_index,best_Split_Point,bestGini=chooseBestFeatureToSplit(Dataset)
print(labels[bestFeature_index])
print(best_Split_Point)
print(bestGini)

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

# 递归构造决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 递归停止的第一个条件：当前集合所有样本标签相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归停止的第二个条件：遍历完所有特征，样本仍未被分“纯”
    # 使用majority_label，返回最多样本数的那个标签的值
    if len(dataSet[0]) == 1:
        return majority_label(classList)
    # 下面是正式建树的过程
    # 选取进行分支的最佳特征的下标和最佳切分点
    bestFeature_index, best_Split_Point, bestGini= chooseBestFeatureToSplit(dataSet)
    # 得到最佳特征
    best_feature = labels[bestFeature_index]
    # 初始化决策树
    decision_tree = {best_feature: {}}
    # 使用过当前最佳特征后将其删去
    del (labels[bestFeature_index])
    # 子特征 = 当前特征（因为刚才已经删去了用过的特征）
    sub_labels = labels[:]
    # 递归调用create_decision_tree去生成新节点
    # 生成由最优切分点划分出来的二分子集
    sub_dataset1, sub_dataset2 = create_sub_dataset(dataSet, bestFeature_index, best_Split_Point)
    # 构造左子树
    decision_tree[best_feature][best_Split_Point] = createTree(sub_dataset1, sub_labels)
    # 构造右子树
    decision_tree[best_feature]['others'] = createTree(sub_dataset2, sub_labels)
    return decision_tree

#测试
dataset, features = createDataSet()
decision_tree = createTree(dataset, features)
print(decision_tree)
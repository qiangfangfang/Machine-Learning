import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
import matplotlib

#生成所有测试样本点
def make_meshgrid(x,y,h=.02):
    x_min,x_max = x.min()-1,x.max()+1
    y_min,y_max = y.min() - 1,y.max() + 1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    return xx,yy


# 对测试样本进行预测，并显示
def plot_test_results(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)

# 载入iris数据集
iris = datasets.load_iris()
# 只使用前面连个特征
X = iris.data[:, :2]
# 样本标签值
y = iris.target

# 创建并训练决策树
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)

title = ('DecisionTreeClassifier')

fig, ax = plt.subplots(figsize = (5, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
# 生成所有测试样本点
xx, yy = make_meshgrid(X0, X1)

# 显示测试样本的分类结果
plot_test_results(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# 显示训练样本
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()

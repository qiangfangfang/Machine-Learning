import pandas as pd
iris = pd.read_csv("iris.csv")
print(iris.head(10))

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(iris,hue = "Species")
plt.show()



print(iris["Species"].value_counts())

# 训练集测试集划分
from sklearn.model_selection import train_test_split
X = iris[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]]
y = iris["Species"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

#模型训练
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression(C=1e3,solver='lbfgs',max_iter=1000)
classifier.fit(X_train.values,y_train)

# 模型评估 使用metrics性能评估
from sklearn import metrics
predict_y = classifier.predict(X_test.values)
#X_train和X_test是带有特征名称的，在输入模型时候只需输入数值，因此用**.values代替即可消除警告
print(metrics.classification_report(y_test,predict_y))

# 新的数据预测
print(classifier.predict([[5.1,3.5,0.4,1.2]]))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification

class logistic_regression():
    def __init__(self):
        pass

    # 激活函数
    def sigmoid(self,x):
        z = 1/(1+np.exp(-x))
        return z

    # 参数w和b的初始化
    def initialize_params(self,dims):
        w = np.zeros((dims,1))
        b = 0
        return w,b

    # y=1的概率，h_theta
    def hypothetic(self,X,w,b):
        h = self.sigmoid(np.dot(X,w)+b)
        return h

    # 极大似然估计法导出交叉熵损失函数
    def loss(self,X,y,w,b):
        h = self.hypothetic(X,w,b)
        num_train = X.shape[0]
        loglikelihood =np.sum(y*np.log(h)+(1-y)*np.log(1-h))
        loss_func= -1/num_train * loglikelihood
        return loss_func

    # 损失函数关于w和b求偏导
    def partial_params(self,X,y,w,b):
        num_train = X.shape[0]
        h = self.hypothetic(X, w, b)
        dw = np.dot(X.T, (h - y)) / num_train
        db = np.sum(h-y)/num_train
        return dw,db

    #梯度下降法估计模型参数
    def gradient_descent(self,X,y,lr,epochs):
        w, b = self.initialize_params(X.shape[1])
        cost_list = []
        for i in range(epochs):
            cost = self.loss(X,y,w,b)
            dw, db =self.partial_params(X,y,w,b)
            w = w - lr * dw
            b = b - lr * db
            if i%100 ==0:
                cost_list.append(cost)
                print('epoch %d cost %f'%(i,cost))
        params = {
            'w':w,
            'b':b
        }

        grads = {
            'dw':dw,
            'db':db
        }
        return cost_list,params,grads

    # 根据学得模型参数预测
    def predict(self,X,params):
        y_pred = self.sigmoid(np.dot(X,params['w'])+params['b'])
        y_pred[y_pred>0.5]=1
        y_pred[y_pred <= 0.5] = 0
        return y_pred

    #计算准确率
    def accuracy(self,y_test,y_pred):
        accuracy_score = np.sum(y_test == y_pred) / len(y_test)
        return accuracy_score

    # 训练集测试集数据产生
    def creat_data(self):
        X, labels = make_classification(n_samples=400, n_features=2,
                                        n_redundant=0,random_state=1)
        labels=labels.reshape((-1,1))
        offset = int(X.shape[0]*0.7)  #70%作为训练集
        X_train,y_train=X[:offset],labels[:offset]
        X_test,y_test = X[offset:],labels[offset:]
        return X_train,y_train,X_test,y_test

    #绘图设置
    def plot_logistic(self,X_train,y_train,params):
        n = X_train.shape[0]
        xcord1=[]
        ycord1=[]
        xcord2=[]
        ycord2=[]
        for i in range(n):
            if y_train[i]==1:
                xcord1.append(X_train[i][0])
                ycord1.append(X_train[i][1])
            else:
                xcord2.append(X_train[i][0])
                ycord2.append(X_train[i][1])
        fig =plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(xcord1,ycord1,s=32,c='red')
        ax.scatter(xcord2,ycord2,s=32,c='blue')
        x=np.arange(-3,3,0.1)
        y=(-params['b']-params['w'][0]*x)/params['w'][1]   #线性决策边界
        ax.plot(x,y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.ylim(-6,8)
        plt.show()

if __name__=="__main__":
    model = logistic_regression()
    X_train,y_train,X_test,y_test=model.creat_data()
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    cost_list, params, grads = model.gradient_descent(X_train, y_train, 0.01, 1000)
    print(params)
    y_train_pred = model.predict(X_train,params)
    accuracy_score_train = model.accuracy(y_train,y_train_pred)
    print('train accuracy is :',accuracy_score_train)

    y_test_pred = model.predict(X_test,params)
    accuracy_score_test=model.accuracy(y_test,y_test_pred)
    print('test accuracy is:',accuracy_score_test)

    model.plot_logistic(X_train,y_train,params)

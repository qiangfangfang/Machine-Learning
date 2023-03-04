import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification

class logistic_regression():
    def __init__(self):
        pass

    def sigmoid(self,x):
        z = 1/(1+np.exp(-x))
        return z

    def initialize_params(self,dims):
        w = np.zeros((dims,1))
        b = 0
        return w,b


    def logistic(self,X,y,w,b):
        num_train = X.shape[0]
        num_feature = X.shape[1]
        h = self.sigmoid(np.dot(X,w)+b)
        cost = -1/num_train * np.sum(y*np.log(h)+(1-y)*np.log(1-h))

        dw = np.dot(X.T,(h-y))/num_train
        db = np.sum(h-y)/num_train
        cost = np.squeeze(cost)
        return h,cost,dw,db

    def logistic_train(self,X,y,lr,epochs):
        w,b = self.initialize_params(X.shape[1])
        cost_list = []
        for i in range(epochs):
            h,cost,dw,db = self.logistic(X,y,w,b)
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

    def predict(self,X,params):
        y_pred = self.sigmoid(np.dot(X,params['w'])+params['b'])
        for i in range(len(y_pred)):
            if y_pred[i]>0.5:
                y_pred[i]=1
            else:
                y_pred[i]=0
        return y_pred

    def accuracy(self,y_test,y_pred):
        correct_count = 0
        for i in range(len(y_test)):
            if y_test[i]==y_pred[i]:
                correct_count +=1
        accuracy_score = correct_count/len(y_test)
        return accuracy_score

    def creat_data(self):
        X,labels = make_classification(n_samples=400,n_features=2,n_redundant=0,
                                       n_informative=2,random_state=1,n_clusters_per_class=2)
        labels=labels.reshape((-1,1))
        offset = int(X.shape[0]*0.7)
        X_train,y_train=X[:offset],labels[:offset]
        X_test,y_test = X[offset:],labels[offset:]
        return X_train,y_train,X_test,y_test

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
        y=(-params['b']-params['w'][0]*x)/params['w'][1]
        ax.plot(x,y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.ylim(-6,8)
        plt.show()

if __name__=="__main__":
    model = logistic_regression()
    X_train,y_train,X_test,y_test=model.creat_data()
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    cost_list,params,grads=model.logistic_train(X_train,y_train,0.01,1000)
    print(params)
    y_train_pred = model.predict(X_train,params)
    accuracy_score_train = model.accuracy(y_train,y_train_pred)
    print('train accuracy is :',accuracy_score_train)

    y_test_pred = model.predict(X_test,params)
    accuracy_score_test=model.accuracy(y_test,y_test_pred)
    print('test accuracy is:',accuracy_score_test)

    model.plot_logistic(X_train,y_train,params)

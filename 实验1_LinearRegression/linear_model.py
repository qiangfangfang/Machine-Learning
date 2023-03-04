import numpy as np

# 加载样本数据x y


x= np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
y=np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

test=np.array([128.15, 45.00, 141.43, 106.27, 99.00, 53.84, 85.36, 70.00])

# #解析解公式计算模型参数
# mean_x=np.mean(x)
# mean_y=np.mean(y)
#
# sum_xy=np.sum((x-mean_x)*(y-mean_y))
# sum_xx=np.sum((x-mean_x)*(x-mean_x))
#
# w=sum_xy/sum_xx
# b=mean_y-w*mean_x
#
# print("w=",w)
# print("b=",b)
#
# #预测房价
# y_pred= w * test + b
# print("预测房价：",y_pred)

# #
#
# from sklearn.linear_model import LinearRegression
# x=x.reshape(-1, 1)
# y=y.reshape(-1, 1)
# test=test.reshape(-1,1)
# model = LinearRegression()   #线性回归类
# model.fit(x,y)
# w=model.coef_
# b=model.intercept_
# print("w=",w)
# print("b=",b)
# print("预测房价" , model.predict(test))

# 梯度下降法估计参数
import sys
from matplotlib import pyplot as plt

lr=0.00001
iter = 100
display = 10

w=np.random.randn()
b=np.random.randn()

mse = []

for i in range(0,iter+1):
    dl_dw = np.mean(x*(w*x+b-y))
    dl_db = np.mean(w*x+b-y)

    w=w-lr*dl_dw
    b=b-lr*dl_db

    pred = w*x+b
    loss = np.mean(np.square(y-pred))/2
    mse.append(loss)

    plt.plot(x,pred)

    if i % display ==0:
        print("i:%i,Loss:%f,w:%f,b:%f"%(i,mse[i],w,b))

plt.figure()
plt.scatter(x,y,color = "red",label = "Real_Price")
plt.plot(x,pred,color="blue",label = "Gradient descent")
plt.plot(x,0.89*x+5.41,color = "green",label = "Analytical solution")
plt.xlabel("Area",fontsize = 14)
plt.ylabel("Price",fontsize = 14)
plt.legend(loc = "upper left")
plt.show()
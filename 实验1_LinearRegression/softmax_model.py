import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

class softmax_regression():
    def __init__(self):
        pass


    def load_dataset(self,file_path):
        dataMat = []
        labelMat = []
        fr = open(file_path)
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat


    def train(self,data_arr, label_arr, n_class, iters=1000, alpha=0.1, lam=0.01):
        '''
        @description: softmax 训练函数
        @param {type}
        @return: theta 参数
        '''
        n_samples, n_features = data_arr.shape
        n_classes = n_class
        # 随机初始化权重矩阵
        weights = np.random.rand(n_class, n_features)
        # 定义损失结果
        all_loss = list()
        # 计算 one-hot 矩阵
        y_one_hot = self.one_hot(label_arr, n_samples, n_classes)
        for i in range(iters):
            # 计算 m * k 的分数矩阵
            scores = np.dot(data_arr, weights.T)
            # 计算 softmax 的值
            probs = self.softmax(scores)
            # 计算损失函数值
            loss = - (1.0 / n_samples) * np.sum(y_one_hot * np.log(probs))
            all_loss.append(loss)
            # 求解梯度
            dw = -(1.0 / n_samples) * np.dot((y_one_hot - probs).T, data_arr) + lam * weights
            dw[:, 0] = dw[:, 0] - lam * weights[:, 0]
            # 更新权重矩阵
            weights = weights - alpha * dw
        return weights, all_loss


    def softmax(self,scores):
        # 计算总和
        sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = np.exp(scores) / sum_exp
        return softmax


    def one_hot(self,label_arr, n_samples, n_classes):
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), label_arr.T] = 1
        return one_hot


    def predict(self,test_dataset, label_arr, weights):
        scores = np.dot(test_dataset, weights.T)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1).reshape((-1, 1))


    def accuracy_plot(self):
        data_arr, label_arr = self.load_dataset('train_dataset.txt')
        data_arr = np.array(data_arr)
        label_arr = np.array(label_arr).reshape((-1, 1))
        weights, all_loss = self.train(data_arr, label_arr, n_class=4)

        # 计算预测的准确率
        test_data_arr, test_label_arr = self.load_dataset('test_dataset.txt')
        test_data_arr = np.array(test_data_arr)
        test_label_arr = np.array(test_label_arr).reshape((-1, 1))
        n_test_samples = test_data_arr.shape[0]
        y_predict = self.predict(test_data_arr, test_label_arr, weights)
        accuray = np.sum(y_predict == test_label_arr) / n_test_samples
        print(accuray)

        # 绘制损失函数
        fig = plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1000), all_loss)
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":

    #
    np.random.seed(13)
    X, y = make_blobs(centers=4, n_samples=5000)
    # 绘制数据分布
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Dataset")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")

    y = y.reshape((-1, 1))
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_dataset = np.append(X_train, y_train, axis=1)
    test_dataset = np.append(X_test, y_test, axis=1)
    np.savetxt("train_dataset.txt", train_dataset, fmt="%.4f %.4f %d")
    np.savetxt("test_dataset.txt", test_dataset, fmt="%.4f %.4f %d")

    model=softmax_regression()
    model.accuracy_plot()



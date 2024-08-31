import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

lr = 0.03
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds
def BP(x, y, units):
    w1 = np.random.rand(x.shape[1], units)
    ww1 = np.random.rand(x.shape[1], units)  # 用ww1表示w1的梯度矩阵，下同
    b = np.random.rand(1, units)
    b1 = np.random.rand(1, units)
    w2 = np.random.rand(units, y.shape[1])
    ww2 = np.random.rand(units, y.shape[1])
    epochs = 1000
    mse = np.ones((1, epochs))
    for m in range(epochs):
        u = np.dot(x, w1) + b
        y_pre = np.dot(sigmoid(u), w2)
        print('After epochs %d, the predicted value is : %4f  %4f  %4f' % (m, y_pre[0, 0], y_pre[0, 1], y_pre[0, 2]))
        mse[0, m] = np.mean(np.square(d_t - y_pre))
        # 首先计算w2的梯度
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                ww2[i, j] = -2 / d_t.shape[1] * (d_t[0, j] - y_pre[0, j]) * sigmoid(u[0, i])
        # 再计算w1的梯度
        for p in range(w1.shape[0]):
            for q in range(w2.shape[1]):
                ww1[p, q] = -2 / d_t.shape[1] * np.sum((w2[q, :] * (d_t[0, :] - y_pre[0, :]))) * sigmoid(
                    u[0, q]) * (
                                    1 - sigmoid(u[0, q])) * x[0, p]
        # 最后计算偏置b的梯度
        for t in range(b.shape[1]):
            b1[0, t] = -2 / d_t.shape[1] * np.sum((w2[q, :] * (d_t[0, :] - y_pre[0, :]))) * 1
        # 更新参数
        w1 = w1 - lr * ww1
        w2 = w2 - lr * ww2
        b = b - lr * b1
    plt.plot(np.arange(0, epochs, 1), mse[0, :])
    plt.show()

df1=pd.read_excel('C:/Users/HP/Downloads/Bp神经网络/Bp神经网络/1.xlsx',0)
df1=df1.iloc[:,:]
#进行数据归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

Inum = x.shape[1]
Hnum = 30
Onum = 1

m, n = x.shape

#划分训练集测试集
cut=45#取最后60行为测试集
m = m-cut
x_t, x_gen=x.iloc[:-cut],x.iloc[-cut:]#列表的切片操作，X.iloc[0:2400，0:7]即为1-2400行，1-7列
d_t, d_gen=y.iloc[:-cut],y.iloc[-cut:]

x_t, d_t=x_t.values, d_t.values.reshape(-1,1)
x_gen, d_gen=x_gen.values, d_gen.values.reshape(-1,1)

BP(x_t, d_t,30)
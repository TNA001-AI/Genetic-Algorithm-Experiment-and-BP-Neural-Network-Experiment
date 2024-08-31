import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def fun_s(x):
    return 1.0 / (1.0 + np.exp(-x))


df1 = pd.read_excel('C:/Users/HP/Desktop/人工智能实验/bp_python/1.xlsx', 0)
df1 = df1.iloc[:, :]
# 进行数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
df0 = min_max_scaler.fit_transform(df1)  # 将训练与测试集数据归一化
df = pd.DataFrame(df0, columns=df1.columns)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

Inum = x.shape[1]
Hnum = 31
Onum = 1

m, n = x.shape

# 划分训练集测试集
cut = 45  # 取最后60行为测试集 86
m = m-cut
# 列表的切片操作，X.iloc[0:2400，0:7]即为1-2400行，1-7列
x_t, x_gen = x.iloc[:-cut], x.iloc[-cut:]
d_t, d_gen = y.iloc[:-cut], y.iloc[-cut:]

x_t, d_t = x_t.values, d_t.values.reshape(-1, 1)
x_gen, d_gen = x_gen.values, d_gen.values.reshape(-1, 1)


# Initialize weights
Wih = 2 * np.random.rand(Inum, Hnum)-1
Who = 2 * np.random.rand(Hnum, Onum)-1
# Wih = np.ones((Inum, Hnum))
# Who = np.ones((Hnum, Onum))
dw_wih = np.zeros((Inum, Hnum))
dw_who = np.zeros((Hnum, Onum))

# Train network
E_max = 1e-4
Train_num = 1000
eta = 60
aerf = 0.4
E = []
plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.886)

for i in range(Train_num):
    Hin = np.dot(x_t, Wih)
    Hout = fun_s(Hin)
    Opin = np.dot(Hout, Who)
    Opot = fun_s(Opin)
    E_p = (d_t - Opot)
    E_train = np.sum(0.5 * E_p ** 2) / m
    E.append(E_train)

    if i % 20 == 0:
        plt.subplot(3, 1, 1)
        plt.title("Training Error", fontsize='15')  # 添加标题
        plt.xlabel("Epochs \n (a)")
        plt.ylabel("Error")
        plt.plot(E[:i], 'r--')
        plt.pause(0.00001)
        plt.draw()

    if E_train < E_max:
        flag = 1
        break

    detea_ho = Opot * (1 - Opot) * E_p
    
    dw_ho = np.zeros((Hnum, m))
    for j in range(m):
        tmp_Hout = Hout[[j]].T
        dw_ho[:, j, np.newaxis] = eta * detea_ho[j] * tmp_Hout
    dw_who = np.mean(dw_ho, axis=1, keepdims=True) + aerf * dw_who

    detea_ih = np.zeros((m, Hnum))
    for j in range(m):
        detea_ih[j] = Hout[j, :] * (1 - Hout[j, :]) * (detea_ho[j] * Who.T)
        
    dw_ih = np.zeros((Inum, Hnum, m))
    for j in range(m):
        for k in range(Hnum):
            tmp_x_t = x_t[[j]].T
            dw_ih[:, k, j, np.newaxis] = eta * detea_ih[j, k] * tmp_x_t
    dw_wih = np.mean(dw_ih, axis=2) + aerf * dw_wih
    Wih = Wih + dw_wih  # 更新权重
    Who = Who + dw_who
Hin_train = np.dot(x_t, Wih)
Hout_train = fun_s(Hin_train)
Opin_train = np.dot(Hout_train, Who)
Opot_train = fun_s(Opin_train)

# plt.figure()
# E_r = abs(d_t - Opot_train)
# plt.plot(E_r, linewidth=1, linestyle="solid")

# # plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
# plt.title("Training Error",fontsize='20') #添加标题


plt.subplot(3, 1, 2)
plt.plot(d_t, linewidth=1, linestyle="solid")
plt.plot(Opot_train, linewidth=1, linestyle="solid")
plt.ylabel("Output")
plt.xlabel("Samples \n (b) ")
plt.legend(('real', 'predict'), loc='upper right', fontsize='10')
plt.title("Train Outputs ", fontsize='15')  # 添加标题


Hin_train = np.dot(x_gen, Wih)
Hout_train = fun_s(Hin_train)
Opin_train = np.dot(Hout_train, Who)
Opot_train = fun_s(Opin_train)

plt.subplot(3, 1, 3)
plt.plot(d_gen, linewidth=1, linestyle="solid")
plt.plot(Opot_train, linewidth=1, linestyle="solid")
plt.ylabel("Output")
plt.xlabel("Samples \n (c)")
plt.legend(('real', 'predict'), loc='upper right', fontsize='10')
plt.title("Test Outputs ", fontsize='15')  # 添加标题
plt.show()

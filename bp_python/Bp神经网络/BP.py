# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BPNN
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#导入必要的库
df1=pd.read_excel('C:/Users/HP/Downloads/Bp神经网络/Bp神经网络/1.xlsx',0)
df1=df1.iloc[:,:]
#进行数据归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
#划分训练集测试集
cut=45#取最后60行为测试集
x_train, x_test=x.iloc[:-cut],x.iloc[-cut:]#列表的切片操作，X.iloc[0:2400，0:7]即为1-2400行，1-7列
y_train, y_test=y.iloc[:-cut],y.iloc[-cut:]
x_train, x_test=x_train.values, x_test.values
y_train, y_test=y_train.values, y_test.values
#神经网络搭建
bp1 = BPNN.BPNNRegression([15, 1, 1])
train_data = [[sx.reshape(15,1), sy.reshape(1,1)] for sx, sy in zip(x_train, y_train)]
train_data_1= [np.reshape(sx, (15,1)) for sx in x_train]
test_data = [np.reshape(sx, (15,1)) for sx in x_test]
#神经网络训练
bp1.MSGD(train_data, 1000, len(train_data), 0.5)
#神经网络预测

y_predict=bp1.predict(test_data)
y_pre = np.array(y_predict)  # 列表转数组
y_pre=y_pre.reshape(cut,1)
y_pre=y_pre[:,0]

y_train_pre = bp1.predict(train_data_1)
y_pre_tr = np.array(y_train_pre)
y_pre_tr = y_pre_tr.reshape(90,1)
y_pre_tr = y_pre_tr[:,0]
#画图 #展示在训练，测试集上的表现
draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_pre_tr)],axis=1);
# draw.iloc[:,0].plot(figsize=
#                     (12,6))
# draw.iloc[:,1].plot(figsize=(12,6))
# plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
# plt.title("Train Data",fontsize='30') #添加标题
# plt.show()

draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pre)],axis=1);
# draw.iloc[:,0].plot(figsize=
#                     (12,6))
# draw.iloc[:,1].plot(figsize=(12,6))

plt.figure()



plt.plot(y_train, linewidth=1, linestyle="solid")
plt.plot(y_pre_tr, linewidth=1, linestyle="solid")
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
plt.show()

plt.figure()


plt.plot(y_test, linewidth=1, linestyle="solid")
plt.plot(y_pre, linewidth=1, linestyle="solid")
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
plt.show()
#输出精度指标
print('测试集上的MAE/MSE')
print(mean_absolute_error(y_pre, y_test))
print(mean_squared_error(y_pre, y_test) )
mape = np.mean(np.abs((y_pre-y_test)/(y_test)))*100
print('=============mape==============')
print(mape,'%')
# 画出真实数据和预测数据的对比曲线图
print("R2 = ",metrics.r2_score(y_test, y_pre)) # R2



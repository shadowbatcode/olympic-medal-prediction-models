import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats as stats

# 读取Excel文件中的数据
file_path = '问题1.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel表的前三列是自变量，第四列是因变量
# 如果不是这样，请相应地调整列索引
X = df.iloc[:, :3].values  # 前三列作为自变量X
y = df.iloc[:, 3].values   # 第四列作为因变量y

X_train = X
y_train = y

# 创建线性回归模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测（这里我们使用X_train中的数据进行预测，但通常你会对新数据进行预测）
# 为了演示置信区间，我们仍然使用X_train中的部分数据点
X_test = X_train[:2]  # 选择前两个数据点进行预测
y_pred = model.predict(X_test)

# 计算训练集的均方误差（MSE）
y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)

# 注意：以下计算置信区间的方法是基于一些简化的假设
# 在实际应用中，你可能需要使用更复杂的方法来考虑预测误差的协方差

# 计算预测的标准误差（这里我们简化为一个标量，但通常这应该是一个与y_pred形状相同的数组）
# 由于我们没有预测误差的协方差矩阵，这里只能做一个简化的估计
# 一个更准确的做法是使用交叉验证或bootstrap来估计预测误差的方差
se_scalar = np.sqrt(mse)  # 简化为标量

# 置信水平
confidence = 0.95
z = stats.norm.ppf(1 - (1 - confidence) / 2)  # 计算z值

# 计算置信区间（使用简化的标量se_scalar）
ci = z * se_scalar
y_pred_lower = y_pred - ci
y_pred_upper = y_pred + ci

# 打印结果
print("预测值:")
print(y_pred)
print("置信区间 ({}%):".format(confidence * 100))
print("Lower Bound:", y_pred_lower)
print("Upper Bound:", y_pred_upper)


#问题2的预测，假设问题2的数据在sheet2中,读取Excel文件中的数据
file_path = '问题1.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet2'  # 替换为你的工作表名称
df2 = pd.read_excel(file_path, sheet_name=sheet_name)
X_2 = df2.iloc[1,:].values
y_pred2 = model.predict(X_2)
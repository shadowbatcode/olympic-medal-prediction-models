import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.utils import resample
from scipy.stats import norm

# 读取数据（假设数据在Excel文件中，前三列为特征，第四列为目标变量）
file_path = '问题1.2.xlsx'
df = pd.read_excel(file_path)
X = df.iloc[:, :3].values  # 前三列作为自变量X
y = df.iloc[:, 3].values   # 第四列作为因变量y

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为DMatrix格式（XGBoost的输入格式）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 6,  # 树的最大深度
    'eta': 0.1,  # 学习率
    'eval_metric': 'rmse'  # 评估指标：均方根误差
}

# Bootstrap参数
num_bootstraps = 1000  # Bootstrap样本的数量
bootstrap_size = len(X_train)  # 每个Bootstrap样本的大小（与训练集相同）

# 存储Bootstrap预测结果的列表
y_pred_bootstraps = []

# Bootstrap循环
for _ in range(num_bootstraps):
    # 有放回地抽样创建Bootstrap样本
    X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, replace=True, n_samples=bootstrap_size,
                                                    random_state=np.random.randint(0, 1e9))

    # 将Bootstrap样本转换为DMatrix格式
    dtrain_bootstrap = xgb.DMatrix(X_train_bootstrap, label=y_train_bootstrap)

    # 训练XGBoost模型
    bst = xgb.train(params, dtrain_bootstrap, num_boost_round=100)  # 注意：这里应该使用交叉验证或早停来避免过拟合，但为了简化示例，我们使用了固定的迭代次数

    # 对测试集进行预测
    y_pred_bootstrap = bst.predict(dtest)

    # 存储预测结果
    y_pred_bootstraps.append(y_pred_bootstrap)

# 将预测结果转换为NumPy数组，以便后续处理
y_pred_bootstraps = np.array(y_pred_bootstraps)

# 计算预测区间的界限（例如，95%的预测区间）
lower_bounds = np.percentile(y_pred_bootstraps, 2.5, axis=0)
upper_bounds = np.percentile(y_pred_bootstraps, 97.5, axis=0)

# 输出预测区间和预测值（可选）
for i in range(len(y_test)):
    print(f"True value: {y_test.iloc[i]}, Predicted value: {np.mean(y_pred_bootstraps[:, i])}, "
          f"95% Prediction Interval: [{lower_bounds[i]}, {upper_bounds[i]}]")

# 注意：这个示例代码为了简化而省略了一些重要的步骤，比如交叉验证、模型调优和早停等。
# 在实际应用中，你应该考虑这些步骤以获得更好的模型性能和更可靠的预测区间。
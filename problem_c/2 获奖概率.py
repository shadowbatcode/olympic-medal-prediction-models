
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from utils import *

athletes = pd.read_csv('summerOly_athletes.csv', encoding='ISO-8859-1')
medal = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')
medal['NOC'] = medal['NOC'].apply(clean_text).apply(normalize_text)
athletes['NOC'] = athletes['NOC'].apply(noc_to_country)
medal_counts_df = medal.groupby('NOC').sum().reset_index().drop(columns=['Year','Rank'])
# 筛选出未获得金牌的国家
countries_no_gold = medal_counts_df[medal_counts_df['Gold'] == 0]['NOC'].unique()
# 筛选出这些国家的历年数据
medal_counts = medal_counts_df[medal_counts_df['NOC'].isin(countries_no_gold)].copy()
athlete_summary = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Athletes').drop(columns='Year')
# 合并数据
data = pd.merge(medal_counts, athlete_summary, on='NOC', how='left')
data = data.fillna(0)  # 填充缺失值

# 计算历史奖牌特征
data['Past_Gold_Avg'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Total_Avg'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Gold_Sum'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())
data['Past_Total_Sum'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())

# 填充历史特征的缺失值
data[['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']] = data[
    ['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']].fillna(0)

# 特征选择（移除 'Time' 和 'Athletes'）
features = ['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']
target_gold = 'Gold'
target_total = 'Total'

# 数据标准化
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])
x = data[features]
y = data[target_gold]
X_train, X_test, y_train_gold, y_test_gold = train_test_split(x, y, test_size=0.2, random_state=42)


# 对特征进行标准化（如果需要的话）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_gold_classified = (y_train_gold > 0).astype(int)
y_test_gold_classified = (y_test_gold > 0).astype(int)
# 定义集成的模型列表
models = [LogisticRegression(penalty='l1', solver='liblinear'),
          ElasticNet(alpha=0.1, l1_ratio=0.7),
          RandomForestClassifier(),
          XGBClassifier()]

# 存储预测结果
predictions_gold = np.zeros((X_test.shape[0], len(models)))
# 模型训练和预测
for i, model in enumerate(models):
    model.fit(X_train_scaled, y_train_gold)
    predictions_gold[:, i] = model.predict(X_test_scaled)  # 预测金牌数


# 计算每个模型的平均预测值（集成结果）
predicted_gold_avg = predictions_gold.mean(axis=1)
print('Gold Avg:', predicted_gold_avg)
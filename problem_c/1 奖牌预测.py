import pandas as pd
import numpy as np
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
import shap
shap.initjs()
# 加载数据，指定编码为 ISO-8859-1
athletes = pd.read_csv('summerOly_athletes.csv', encoding='ISO-8859-1')
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')
host_countries = pd.read_csv('summerOly_hosts.csv', encoding='ISO-8859-1')
country_feature = pd.read_csv('GDP_data.csv')

athletes['NOC'] = athletes['NOC'].apply(noc_to_country)
medal_counts['NOC'] = medal_counts['NOC'].apply(clean_text).apply(normalize_text)
host_countries['Host'] = host_countries['Host'].apply(clean_text).apply(normalize_text).apply(process_host_data)
medal_summary = medal_counts.groupby(['Year', 'NOC']).agg({'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum', 'Total': 'sum','Rank':'mean'}).reset_index()
athlete_summary = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Athletes')


def add_host_data(row,host_countries):
    chosen_host = host_countries[host_countries['Year'] == row['Year']]
    if row['NOC'] == chosen_host['Host'].values:
        row['Host'] = 1
        return row
    else:
        row['Host'] = 0
        return row


# 3. 合并数据
data = pd.merge(medal_summary, athlete_summary, on=['Year', 'NOC'], how='left')
data = pd.merge(data, country_feature, on=['Year', 'NOC'], how='left')
data = data.apply(lambda row: add_host_data(row, host_countries), axis=1)
data = data.fillna(0)  # 填充缺失值

average_gdp_per_country = data.groupby('NOC')['GDP'].mean()
def fill_with_average(row):
    if row['GDP'] == 0:
        return average_gdp_per_country[row['NOC']]
    else:
        return row['GDP']
data['GDP'] = data.apply(fill_with_average, axis=1)






# 4. 添加历史奖牌特征
# 计算每个国家过去3届奥运会的平均奖牌数和总奖牌数
data['Past_Gold_Avg'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Total_Avg'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Gold_Sum'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())
data['Past_Total_Sum'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())

# 填充历史特征的缺失值（对于早期数据）
data[['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']] = data[
    ['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']].fillna(0)


features = ['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum','Silver','Bronze','Rank', 'Athletes', 'GDP','Host']
target_gold = 'Gold'
target_total = 'Total'

# 定义聚类的数量 c 和模糊因子 m
c = 5  # 假设有5个不同的国家群体
m = 2  # 模糊因子，通常取值大于1
error = 0.005  # 收敛误差阈值
maxiter = 1000  # 最大迭代次数

# 对每个NOC的特征进行聚合处理
aggregated_data = data.groupby('NOC')[features].mean().reset_index()
# 标准化特征数据
X = aggregated_data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 执行模糊C均值聚类
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled.T, c, m, error, maxiter, init=None)
# 获取每个样本点的最大隶属度对应的簇标签
cluster_membership = np.argmax(u, axis=0)
aggregated_data['Cluster'] = cluster_membership
# 合并回原始数据
data = data.merge(aggregated_data[['NOC', 'Cluster']], on='NOC', how='left')
data = data.sort_values("NOC")
# 根据Cluster重新计算每个Cluster的统计信息
group_stats = data.groupby('Cluster')[features].mean()
data.to_csv('Clustered_Data.csv', index=False)
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Gold', y='Silver', hue='Cluster', data=data, palette='viridis')
plt.title('Clusters Based on Gold and Silver Medals')
plt.show()
# 获取所有唯一的NOC
unique_nocs = data['NOC'].unique()
# 获取2024年的平均数据并处理缺失值
mean_2024 = data[data['Year'] == 2024].groupby('NOC').mean().reindex(unique_nocs).fillna(0)
pct_change_2024 = data[data['Year'] == 2024].groupby('NOC').pct_change().reindex(unique_nocs).fillna(0)
diff_2024 = data[data['Year'] == 2024].groupby('NOC').diff().reindex(unique_nocs).fillna(0)

# 构造预测所需的数据框
future_data = pd.DataFrame({
    'Year': 2028,
    'NOC': unique_nocs,
    'Athletes': pct_change_2024['Athletes'].values,
    'Rank': normalize(mean_2024['Rank'].values),
    'GDP': mean_2024['GDP'].values,
    'Host': normalize(mean_2024['Host'].values),
    'Silver': mean_2024['Silver'].values,
    'Bronze': mean_2024['Bronze'].values,
    'Past_Gold_Avg': mean_2024['Past_Gold_Avg'].values,
    'Past_Total_Avg': mean_2024['Past_Total_Avg'].values,
    'Past_Gold_Sum': mean_2024['Past_Gold_Sum'].values,
    'Past_Total_Sum': mean_2024['Past_Total_Sum'].values
})
future_data = future_data.apply(lambda row: add_host_data(row, host_countries), axis=1)
# 存储所有预测结果
all_predictions = []
# 评估模型性能
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

for cluster in range(c):
    subset = data[data['Cluster'] == cluster]
    cluster_nocs = subset['NOC'].unique()
    future_data_cluster = future_data[future_data['NOC'].isin(cluster_nocs)]

    # 对于多个模型，累加其预测结果
    predictions = [];stds = [];evas = []
    for target in [target_gold, target_total]:
        x = subset[features]
        y = subset[target]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model1= LinearRegression()
        model2 = RandomForestRegressor()
        model3 = SVR()
        voting_regressor = VotingRegressor(estimators=[('lr', model1), ('rf', model2), ('svr', model3)])
        voting_regressor.fit(X_train, y_train)
        std = np.std(voting_regressor.predict(subset[features])- subset[target])
        eva =evaluate_model(y_test, voting_regressor.predict(X_test))
        predictions.append(voting_regressor.predict(future_data_cluster[features]))
        stds.append(std);evas.append(eva)
    predictions_gold, predictions_total = predictions
    gold_std, total_std = stds
    avg_mse, avg_r2 = (sum(values)/len(evas) for values in zip(*evas))
    print(f"Cluster {cluster}: Avg MSE={avg_mse:.2f}, Avg R2={avg_r2:.2f}, Gold_std={gold_std:.2f}, Total_std={total_std:.2f}")
    predictions_gold = np.maximum(predictions_gold.clip(min=0).round(), 0).astype(int)
    predictions_total = np.maximum(predictions_total.clip(min=0).round(), 0).astype(int)

    df = pd.DataFrame({
        'NOC': cluster_nocs,
        'Predicted_Gold': predictions_gold,
        'Predicted_Total': predictions_total,
    })
    df['Gold_Lower'] = np.ceil(df['Predicted_Gold'] - 2 * gold_std).clip(lower=0).astype(int)
    df['Gold_Upper'] = np.floor(df['Predicted_Gold'] + 2 * gold_std).clip(lower=0).astype(int)
    df['Total_Lower'] = np.ceil(df['Predicted_Total'] - 2 * total_std).clip(lower=0).astype(int)
    df['Total_Upper'] = np.floor(df['Predicted_Total'] + 2 * total_std).clip(lower=0).astype(int)
    # 将预测结果添加到列表中
    all_predictions.append(df)

# 合并所有预测结果
final_predictions = pd.concat(all_predictions, ignore_index=True)

    # 输出预测结果
future_data = final_predictions.sort_values(by='Predicted_Total', ascending=False)
print("Predicted Medal Table for 2028 Los Angeles Olympics:")
print(
    future_data[['NOC', 'Predicted_Gold', 'Gold_Lower', 'Gold_Upper', 'Predicted_Total', 'Total_Lower', 'Total_Upper']])
future_data.to_csv('Predicted_Medal_Table_2028.csv', index=False)
# 可视化预测的奖牌榜
plt.figure(figsize=(12, 8))
sns.barplot(x='Predicted_Total', y='NOC', data=future_data.head(10), palette='viridis')
plt.title('Predicted Medal Table for 2028 Los Angeles Olympics')
plt.xlabel('Predicted Total Medals')
plt.ylabel('Country')
plt.show()




countries_2024 = data[data['Year'] == 2024]['NOC'].unique()
future_data = future_data[future_data['NOC'].isin(countries_2024)].copy()
future_data = future_data.sort_values('NOC')
future_data['Gold_History'] = data[data['Year'] == 2024]['Gold'].fillna(0).values.round().astype(int)
future_data['Total_History'] = data[data['Year'] == 2024]['Total'].fillna(0).values.round().astype(int)
# 计算进步/退步
future_data['Gold_Progress'] = (future_data['Predicted_Gold'] - future_data['Gold_History']).round().astype(int)
future_data['Total_Progress'] = (future_data['Predicted_Total'] - future_data['Total_History']).round().astype(int)
# 标记进步和退步的国家
future_data['Gold_Trend'] = np.where(future_data['Gold_Progress'] > 0, 'Progress', 'Decline')
future_data['Total_Trend'] = np.where(future_data['Total_Progress'] > 0, 'Progress', 'Decline')
future_data.to_csv('Progress_2028.csv', index=False)
# 按照金牌数的进步排序
# progress_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(by='Gold_Progress', ascending=False)
# decline_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(by='Gold_Progress', ascending=True)
# progress_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(by='Total_Progress', ascending=False)
# decline_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(by='Total_Progress', ascending=True)




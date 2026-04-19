import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, BayesianRidge,ElasticNet
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from utils import *


np.random.seed(42)
country_feature = pd.read_csv('country_feature.csv')
athletes = pd.read_csv('summerOly_athletes.csv', encoding='ISO-8859-1')
medal = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')

medal['NOC'] = medal['NOC'].apply(clean_text).apply(normalize_text)
athletes['NOC'] = athletes['NOC'].apply(noc_to_country)
medal_counts_df = medal.groupby('NOC').sum().reset_index().drop(columns=['Year'])
athlete_summary = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Athletes').drop(columns='Year')

country_gold = medal['NOC'].unique()
gold_data = medal[medal['NOC'].isin(country_gold)].copy()
gold_data.loc[:, 'trend'] = gold_data.groupby('NOC')['Total'].diff().fillna(0)
gold_probabilities = gold_data.groupby('NOC')['trend'].apply(lambda x: np.mean(x > 0))

# 构造训练所需的数据框
data = pd.merge(medal_counts_df, athlete_summary, on='NOC', how='left')
data = pd.merge(data, country_feature, on='NOC', how='left')
data = data.fillna(0)  # 填充缺失值

mean = data.groupby('NOC').mean().reindex(country_gold).fillna(0)
sum_ = data.groupby('NOC').sum().reset_index()
pct_change = data.groupby('NOC').pct_change().reindex(country_gold).fillna(0)
diff = data.groupby('NOC').diff().reindex(country_gold).fillna(0)
# 构造预测所需的数据框
data = pd.DataFrame({
    'NOC': country_gold,
    'Athletes': pct_change['Athletes'].values,
    'Rank': normalize(mean['Rank'].values),
    'Silver': sum_['Silver'].values,
    'Bronze': sum_['Bronze'].values,
    'Gold': normalize(mean['Gold'].values),
    'Total': normalize(gold_probabilities),
})
medal_counts_df = medal.groupby('NOC').sum().reset_index().drop(columns=['Year'])
countries_no_gold = medal_counts_df[medal_counts_df['Gold'] == 0]['NOC'].unique()
no_gold_data = medal[medal['NOC'].isin(countries_no_gold)].copy()
no_gold_data.loc[:, 'trend'] = no_gold_data.groupby('NOC')['Total'].diff().fillna(0)
gold_probabilities = no_gold_data.groupby('NOC')['trend'].apply(lambda x: np.mean(x > 0))


medal_counts = medal_counts_df[medal_counts_df['NOC'].isin(countries_no_gold)].copy()
pre_data = pd.merge(medal_counts, athlete_summary, on='NOC', how='left')
pre_data = pd.merge(pre_data, country_feature, on='NOC', how='left')
pre_data = pre_data.fillna(0)
mean_2024 = pre_data.groupby('NOC').mean().reindex(countries_no_gold).fillna(0)
sum_2024 = pre_data.groupby('NOC').sum().reindex(countries_no_gold).fillna(0)
pct_change_2024 = pre_data.groupby('NOC').pct_change().reindex(countries_no_gold).fillna(0)
diff_2024 = pre_data.groupby('NOC').diff().reindex(countries_no_gold).fillna(0)
# 构造预测所需的数据框
pre_data = pd.DataFrame({
    'NOC': countries_no_gold,
    'Athletes': pct_change_2024['Athletes'].values,
    'Rank': normalize(mean_2024['Rank'].values),
    'Silver': sum_2024['Silver'].values,
    'Bronze': sum_2024['Bronze'].values,
    'Gold': mean_2024['Gold'].values,
    'Total': normalize(gold_probabilities),
})

Train_data = data.drop(columns=['NOC','Gold'])
Train_label = data['Gold']
Pre_data = pre_data.drop(columns=['NOC','Gold'])

X_train, X_test, y_train_gold, y_test_gold = train_test_split(Train_data, Train_label, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_gold_classified = (y_train_gold > 0).astype(int)
y_test_gold_classified = (y_test_gold > 0).astype(int)
Pre_data_scaled = scaler.transform(Pre_data)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
model = BayesianRidge()
model.fit(X_train, y_train_gold_classified)

y_pred = model.predict(X_test)  # 获取预测的类别
model_accuracy = accuracy_score(y_test_gold_classified, y_pred)  # 获取模型的准确率
print('模型准确率：', model_accuracy)
new_country_features = scaler.transform(Pre_data_scaled)
new_country_probability = model.predict(Pre_data_scaled)

probabilities_df = pd.DataFrame({
    'Country': pre_data['NOC'].unique(),
    'Gold_Probability': new_country_probability
})

# 按照金牌概率排序
probabilities_df_sorted = probabilities_df.sort_values(by='Gold_Probability', ascending=False)
probabilities_df_sorted.to_csv('winner_probabilities.csv', index=False)
# 输出排名前几个最有可能获得第一块金牌的国家及其概率
top_n = 10  # 要显示的国家数量
print(f"预计在2028年最有可能获得第一块金牌的国家（前{top_n}名）：")
print(probabilities_df_sorted.head(top_n))

# 输出所有国家的金牌概率，按概率降序排列
print("\n所有国家的金牌预测概率（按概率降序）：")
print(probabilities_df_sorted)
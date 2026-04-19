import pandas as pd

# 加载数据文件 (假设文件名为 'summerOly_medal_counts.csv')
data = pd.read_csv('summerOly_medal_counts.csv')

# 统计每个国家在历届奥运会中没有金牌的次数
no_gold_counts = data[data['Gold'] == 0].groupby('NOC').size()

# 输出没有金牌的国家及其次数
print(no_gold_counts)

# 保存结果到新的CSV文件
no_gold_counts.to_csv('no_gold_counts.csv', header=['No_Gold_Counts'], index=True)

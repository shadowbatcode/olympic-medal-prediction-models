import pandas as pd
import statsmodels.tsa.stattools as ts

# 读取Excel数据
excel_file_path = '问题4.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file_path)

# 确保你的DataFrame包含以下列：'medals'（奖牌数）和'projects'（项目数）
# 如果列名不同，请相应地修改下面的代码

# 将时间序列数据转换为pandas的Series对象
medals = df['medals']
projects = df['projects']

# 为了进行格兰杰因果检验，时间序列数据需要是平稳的。
# 这里我们简单地进行一阶差分来使数据平稳（这只是一个示例，可能不适用于所有情况）。
medals_diff = medals.diff().dropna()
projects_diff = projects.diff().dropna()

# 进行格兰杰因果检验
# maxlag表示滞后阶数的最大值，你可以根据需要调整这个值
maxlag = 2  # 例如，检验滞后1期和滞后2期的情况


granger_test_result = ts.grangercausalitytests(projects_diff.values, medals_diff.values, maxlag=maxlag, verbose=True)

# 打印检验结果
for lag, test_result in granger_test_result.items():
    print(f"Lag: {lag}")
    for key, value in test_result.items():
        print(f"  {key}: {value}")

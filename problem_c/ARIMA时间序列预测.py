import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 读取Excel数据（假设数据在Excel文件的第一列）
file_path = '问题2.xlsx'
df = pd.read_excel(file_path, sheet_name=0, usecols=[0])

# 假设时间序列数据在数据的第一列，我们将其命名为'code'
time_series = df.iloc[:, 0].rename('code')

model = ARIMA(time_series, order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())

# 预测未来一期（注意：这里的forecast()函数返回的是一个包含预测值和置信区间的DataFrame）
forecast = model_fit.forecast(steps=1)
forecast_index = pd.date_range(time_series.index[-1], periods=2, closed='right')[1] if pd.api.types.is_datetime64_dtype(
    time_series.index) else len(time_series)
forecast_series = pd.Series(forecast.iloc[0, 0], index=[forecast_index])
forecast_series = pd.Series(forecast.iloc[0, 0], index=[len(time_series)])

# 输出预测值
print(f"Predicted value for the next period: {forecast_series.iloc[0]}")

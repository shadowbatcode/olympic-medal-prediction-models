import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from main import medal_class_2

def predict_timdata(row, target, steps=1):
    """
    Predict the time data based on the given row data.
    target: the target column name.
    """
    data = row[target].to_numpy()
    result = adfuller(data)
    # stationary test
    if result[1] >= 0.05:
        result = adfuller(np.diff(data))
        print(f"{target} is not stationary, ADF Statistic: {result[0]}, p-value: {result[1]}")

    # AR I MA model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    forecast = model_fit.forecast(steps=steps)

    return forecast

df = pd.DataFrame(columns=['NOC', 'Gold', 'Silver', 'Bronze', 'Total'])
for country in medal_class_2['NOC']:
    forecast_gold = predict_timdata(medal_class_2[medal_class_2['NOC'] == country], 'Gold', steps=5)
    forecast_total = predict_timdata(medal_class_2[medal_class_2['NOC'] == country], 'Total', steps=5)
    print(country)
    print(forecast_gold)
    print(forecast_total)
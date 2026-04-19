import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import shap
from joblib import Parallel, delayed


def preprocess_country_data(df, country):
    """
    数据预处理：过滤数据，处理缺失值，按年份排序。
    """
    country_df = df[df['NOC'] == country].copy()
    country_df.set_index('Year', inplace=True)
    country_df = country_df.sort_index()

    # 填充缺失值
    country_df['GDP'] = country_df['GDP'].fillna(method='ffill').fillna(method='bfill')
    return country_df


def train_model(X_train, y_train):
    """
    训练 XGBoost 模型。
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def forecast_future_gdp(model, forecast_years):
    """
    使用训练好的模型预测未来 GDP。
    """
    future_years = pd.DataFrame({'Year': forecast_years})
    forecast = model.predict(future_years)
    return forecast


def evaluate_model(model, X_test, y_test):
    """
    评估模型的性能。
    """
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse


def visualize_results(country, country_df, forecast, forecast_years):
    """
    可视化实际 GDP 和预测值。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(country_df.index, country_df['GDP'], label='Actual GDP', marker='o')
    plt.plot(forecast_years, forecast, label='Forecasted GDP', linestyle='--', marker='x')
    plt.title(f"GDP Prediction for {country}")
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.legend()
    plt.grid()
    plt.show()


def explain_shap(model, X):
    """
    使用 SHAP 解释模型。
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values.values, X, feature_names=X.columns)
    return shap_values


def process_country(df, country, forecast_years, forecast_data):
    """
    针对单个国家进行数据处理、模型训练、预测和可视化。
    """
    # 数据预处理
    country_df = preprocess_country_data(df, country)

    # 特征与目标变量
    country_df['Year'] = country_df.index
    X = country_df[['Year']]
    y = country_df['GDP']

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 模型训练
    model = train_model(X_train, y_train)

    # 模型评估
    rmse = evaluate_model(model, X_test, y_test)
    print(f"RMSE for {country}: {rmse:.2f}")

    # 未来预测
    forecast = forecast_future_gdp(model, forecast_years)
    forecast_df = pd.DataFrame({'Year': forecast_years, 'GDP': forecast, 'NOC': country})
    forecast_data.append(forecast_df)

    # 可视化
    visualize_results(country, country_df, forecast, forecast_years)

    # SHAP 解释
    explain_shap(model, X)


def main(df):
    """
    主函数：对所有国家进行处理并存储结果。
    """
    forecast_data = []
    all_countries = df['NOC'].unique()
    forecast_years = range(1896,2030)

    # 并行处理每个国家
    Parallel(n_jobs=-1)(
        delayed(process_country)(df, country, forecast_years, forecast_data)
        for country in all_countries
    )

    # 合并所有预测结果
    forecast_df = pd.concat(forecast_data, ignore_index=True)
    forecast_df.to_csv("forecast_results.csv", index=False)
    print("Forecast results saved to 'forecast_results.csv'.")


# 示例调用
if __name__ == "__main__":
    # 假设 df 是一个包含 NOC, Year, GDP 列的数据框
    df = pd.read_csv("GDP_data_cleaned.csv")
    main(df)

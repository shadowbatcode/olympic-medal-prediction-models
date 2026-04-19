#先确保安装了pip install pandas pymc3
import pymc3 as pm
import pandas as pd
import numpy as np

# 读取Excel数据
excel_file_path = '问题3.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file_path)

# Step 2: 构建模型
with pm.Model() as model:
    # 先验分布
    mu_prior = 0
    sigma_prior = 10  # 如果没有先验信息，选择一个较大的值
    beta0 = pm.Normal('beta0', mu=mu_prior, sigma=sigma_prior)
    beta1 = pm.Normal('beta1', mu=mu_prior, sigma=sigma_prior)
    beta2 = pm.Normal('beta2', mu=mu_prior, sigma=sigma_prior)
    beta3 = pm.Normal('beta3', mu=mu_prior, sigma=sigma_prior)

    # 预测变量
    X1 = df['is_host'].values
    X2 = df['athlete_score'].values
    X3 = df['project_score'].values

    # 线性预测器
    mu = beta0 + beta1 * X1 + beta2 * X2 + beta3 * X3

    # 泊松分布
    medals = pm.Poisson('medals', mu=mu, observed=df['medals'].values)

    # Step 3: 拟合模型
    with pm.Sampling(draws=2000, tune=1000, return_inferencedata=True, random_seed=123) as trace:
        pass

# Step 4: 后验推断
print(pm.summary(trace))


# 计算新国家获得奖牌数量的预期值和获得至少一枚奖牌的概率
def predict_medals(is_host, athlete_score, project_score, trace):
    beta0_samples = trace.posterior['beta0'].values
    beta1_samples = trace.posterior['beta1'].values
    beta2_samples = trace.posterior['beta2'].values
    beta3_samples = trace.posterior['beta3'].values

    # 注意：这里我们直接使用了线性预测器的值作为泊松分布的λ参数
    # 但在计算预期值时，我们需要对λ取exp，因为泊松分布的期望是λ
    mu_samples = beta0_samples[:, None] + beta1_samples[:, None] * is_host + beta2_samples[:,
                                                                             None] * athlete_score + beta3_samples[:,
                                                                                                     None] * project_score

    # 预期值
    expected_medals = np.exp(mu_samples).mean(axis=0)

    # 获得至少一枚奖牌的概率
    P_geq_1 = 1 - np.exp(-mu_samples).mean(axis=0)

    return expected_medals, P_geq_1


# 示例：预测一个新国家（是东道国，运动员得分80，项目得分90）
is_host_new = np.array([1])  # 注意这里需要是一个NumPy数组，即使只有一个值
athlete_score_new = np.array([80])
project_score_new = np.array([90])
expected_medals, P_geq_1 = predict_medals(is_host_new, athlete_score_new, project_score_new, trace)

print(f"预期奖牌数量: {expected_medals[0]:.2f}")  # 注意这里需要索引[0]来获取单个国家的预期奖牌数量
print(f"获得至少一枚奖牌的概率: {P_geq_1[0]:.4f}")
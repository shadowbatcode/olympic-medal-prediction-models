# Olympic Medal Prediction Models

该项目围绕奥运会奖牌分布建模展开，目标是结合历史奖牌数据、运动员数据、主办国因素与国家特征变量，对未来奖牌表现进行预测，并分析不同国家在奖牌竞争中的潜在突破机会。

## Project Goals

- 预测未来奥运奖牌榜及各国奖牌变化趋势
- 分析国家层面的进步、退步与突破概率
- 评估零奖牌国家获得奖牌的可能性
- 从多种模型视角比较预测效果与解释性

## Methods

- 线性回归与多元线性回归
- 随机森林、Voting 回归与逻辑回归
- XGBoost 与 bootstrap 区间预测
- ARIMA 时间序列预测
- 泊松回归与格兰杰因果检验
- 统计可视化、地图可视化与三维展示

## Repository Structure

- `problem_c/`
  原始版分析脚本与数据文件，适合查看完整建模过程
- `problem_c_refined/`
  整理后的分析流程，包含 `data/` 数据目录与 `photo/` 图像资源

## Key Scripts

- `problem_c/1 奖牌预测.py`
  奖牌总量预测主脚本
- `problem_c/2 获奖概率.py`
  获奖概率与分类分析
- `problem_c/2-1进步退步分析.py`
  国家表现变化分析
- `problem_c/3-2奖牌突破概率分析.py`
  零奖牌国家突破概率评估
- `problem_c_refined/main.py`
  整理版主流程
- `problem_c_refined/class2.py`
  时间序列分析模块
- `problem_c_refined/class5.py`
  竞争概率计算模块

## Data And Outputs

项目包含奥运历史运动员数据、主办国数据、奖牌统计数据、国家特征数据，以及若干预测结果文件。输出结果主要包括奖牌预测表、概率结果表、进退步分析结果与可视化图像。

## Running

由于脚本大量使用相对路径，建议进入 `problem_c/` 或 `problem_c_refined/` 目录后再执行对应脚本。

## Main Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `statsmodels`
- `shap`
- `pycountry`

# Olympic Medal Prediction Models

来源：
- `2025美赛C/题目C`
- `2025美赛C/题目C整理`

项目内容：
- `problem_c/`
  原始版脚本与 CSV 数据，包含奖牌预测、获奖概率、进步退步分析、ARIMA 分析等
- `problem_c_refined/`
  整理版脚本，包含 `data/` 数据目录和 `photo/` 可视化图片资源

本次整理刻意排除：
- `2025美赛C/.idea`
- `2025美赛C/dataset`
- `2025美赛C/fbprophet-0.7.1`
- `2025美赛C/MathModels`
- `2025美赛C/Test`

原因：
- 上述内容更像 IDE 配置、第三方库、副实验或通用算法包，不适合作为这个主题仓库的根内容

运行提示：
- `problem_c/` 和 `problem_c_refined/` 都大量使用相对路径
- 运行时请先进入对应子目录，再执行脚本

主要依赖：
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `xgboost`
- `statsmodels`
- `pycountry`
- `shap`
- `skfuzzy`
- `geopandas`
- `Pillow`
- `geopy`
- `google_images_download`
- `wbgapi`
- `pymc3`

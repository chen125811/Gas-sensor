# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import json
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 导入其他树模型
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 获取当前脚本的目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "tree_models_results")
os.makedirs(output_dir, exist_ok=True)

# 设置图表样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# 加载数据
print("加载数据...")
df = pd.read_excel("F:\VS code\code for gas adsorption\gas adsorption-C6H6.xlsx")

# 重命名列
df = df.rename(columns={
    "BET Specific surface area": "BET_surface_area", "Pore volume": "Pore_volume",
    "Other active sites": "Other_active_sites", "Open metal sites": "Open_metal_sites",
    "Hydroxyl": "Hydroxyl", "Amino": "Amino", "carboxyl": "Carboxyl",
    "Gas concentration (kPa)": "Gas_concentration", "Measurement method": "Measurement_method",
    "Metal Node": "Metal_Node", "Filling gas": "Filling_gas",
    "Synthesis temperature(℃)": "Synthesis_temperature", "Synthesis time (h)": "Synthesis_time",
    "Adsorption level vs BUT-53": "Adsorption_Level"
})

# 数据清洗
selected_columns = [
    "BET_surface_area", "Pore_volume", "Other_active_sites", "Open_metal_sites", 
    "Hydroxyl", "Amino", "Carboxyl", "Gas_concentration", "Measurement_method",
    "Metal_Node", "Filling_gas", "Synthesis_temperature", "Synthesis_time"
]

# 移除字符串中的星号
def remove_asterisk(value):
    if isinstance(value, str):
        return value.replace("*", "")
    return value

df[selected_columns] = df[selected_columns].applymap(remove_asterisk)

# 处理分类变量
categorical_columns = ["Other_active_sites",  "Metal_Node", 
                       "Measurement_method", "Filling_gas"]
for col in categorical_columns:
    df[col] = df[col].astype(str).factorize()[0]

# 处理二值变量
binary_columns = ["Hydroxyl", "Amino", "Carboxyl", "Open_metal_sites",]
for col in binary_columns:
    df[col] = df[col].replace({'0': 0, '1': 1, 0.0: 0, 1.0: 1})

# 转换为浮点数
df[selected_columns] = df[selected_columns].astype(float)

# 去除缺失值
df_cleaned = df.dropna(subset=selected_columns + ["Adsorption_Level"]).copy()
print(f"数据集大小: {len(df_cleaned)} 条记录")

# 定义变量类型
continuous_vars = ["BET_surface_area", "Pore_volume", "Gas_concentration", 
                  "Synthesis_temperature", "Synthesis_time"]

# 数据准备
X = df_cleaned[selected_columns]
y = df_cleaned["Adsorption_Level"]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_columns)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")

# 变量类型辅助函数
def get_variable_type(feature):
    if feature in continuous_vars:
        return 'Continuous'
    elif feature in binary_columns:
        return 'Binary'
    else:
        return 'Categorical'

# ----- 定义所有模型参数网格 -----
print("\n开始模型参数优化...")

# GBDT参数网格
gbdt_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [1, 3, 5, 7],
    'min_samples_split': [2, 4, 6, 8],
    'subsample': [0.7, 0.8, 0.9, 0.95, 1.0],
    'max_features': [0.5, 0.7, 0.75, 0.9, 1.0]
}

# XGBoost参数网格
xgb_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# RandomForest参数网格
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# LightGBM参数网格
lgb_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9, -1],
    'num_leaves': [20, 31, 50, 100],
    'min_child_samples': [5, 10, 20, 30],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# CatBoost参数网格
cb_param_grid = {
    'iterations': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

# ----- 优化所有模型 -----
optimized_models = {}
model_performances = {}
best_params = {}

# 1. 梯度提升树(GBDT)
print("优化GBDT模型...")
gbdt_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gbdt_param_grid,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
gbdt_search.fit(X_train, y_train)
print(f"GBDT优化用时: {time.time() - start_time:.2f}秒")
optimized_models["GBDT"] = gbdt_search.best_estimator_
best_params["GBDT"] = gbdt_search.best_params_
print(f"GBDT最佳参数: {gbdt_search.best_params_}")
print(f"GBDT最佳交叉验证R²: {gbdt_search.best_score_:.4f}")

# 2. XGBoost
print("\n优化XGBoost模型...")
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_distributions=xgb_param_grid,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
xgb_search.fit(X_train, y_train)
print(f"XGBoost优化用时: {time.time() - start_time:.2f}秒")
optimized_models["XGBoost"] = xgb_search.best_estimator_
best_params["XGBoost"] = xgb_search.best_params_
print(f"XGBoost最佳参数: {xgb_search.best_params_}")
print(f"XGBoost最佳交叉验证R²: {xgb_search.best_score_:.4f}")

# 3. 随机森林
print("\n优化RandomForest模型...")
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
rf_search.fit(X_train, y_train)
print(f"RandomForest优化用时: {time.time() - start_time:.2f}秒")
optimized_models["RandomForest"] = rf_search.best_estimator_
best_params["RandomForest"] = rf_search.best_params_
print(f"RandomForest最佳参数: {rf_search.best_params_}")
print(f"RandomForest最佳交叉验证R²: {rf_search.best_score_:.4f}")

# 4. LightGBM
print("\n优化LightGBM模型...")
lgb_search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1),
    param_distributions=lgb_param_grid,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
lgb_search.fit(X_train, y_train)
print(f"LightGBM优化用时: {time.time() - start_time:.2f}秒")
optimized_models["LightGBM"] = lgb_search.best_estimator_
best_params["LightGBM"] = lgb_search.best_params_
print(f"LightGBM最佳参数: {lgb_search.best_params_}")
print(f"LightGBM最佳交叉验证R²: {lgb_search.best_score_:.4f}")

# 5. CatBoost
print("\n优化CatBoost模型...")
cb_search = RandomizedSearchCV(
    cb.CatBoostRegressor(random_state=42, verbose=0),
    param_distributions=cb_param_grid,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
cb_search.fit(X_train, y_train)
print(f"CatBoost优化用时: {time.time() - start_time:.2f}秒")
optimized_models["CatBoost"] = cb_search.best_estimator_
best_params["CatBoost"] = cb_search.best_params_
print(f"CatBoost最佳参数: {cb_search.best_params_}")
print(f"CatBoost最佳交叉验证R²: {cb_search.best_score_:.4f}")

# 保存最佳参数
with open(os.path.join(output_dir, 'best_model_parameters.json'), 'w', encoding='utf-8') as f:
    json.dump({k: {str(pk): str(pv) for pk, pv in v.items()} 
               for k, v in best_params.items()}, f, indent=4)

# ----- 评估所有模型并比较 -----
print("\n评估所有模型性能...")

results = {}
performance_metrics = []
all_feature_importances = {}

for name, model in optimized_models.items():
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 计算性能指标
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # 记录结果
    results[name] = {
        "model": model,
        "y_pred_test": y_pred_test,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae
    }
    
    performance_metrics.append({
        "Model": name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "RMSE": test_rmse,
        "MAE": test_mae
    })
    
    print(f"\n{name} 模型评估:")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    
    # 提取特征重要性（如果可用）
    if hasattr(model, 'feature_importances_'):
        # 提取并保存特征重要性
        feature_importance_df = pd.DataFrame({
            'Feature': selected_columns,
            'Importance': model.feature_importances_
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        feature_importance_df['Variable_Type'] = feature_importance_df['Feature'].apply(get_variable_type)
        all_feature_importances[name] = feature_importance_df
        
        # 保存为CSV
        feature_importance_df.to_csv(os.path.join(output_dir, f'{name}_feature_importance.csv'), index=False)

# 创建性能比较表并保存
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(os.path.join(output_dir, 'model_performance_comparison.csv'), index=False)

# 找出最佳模型
best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
best_model = results[best_model_name]["model"]
best_model_r2 = results[best_model_name]["test_r2"]
print(f"\n最佳模型是: {best_model_name}，测试集 R² = {best_model_r2:.4f}")

# ----- 可视化模型比较 -----

# 1. 性能条形图比较
plt.figure(figsize=(14, 8))
performance_df_melted = pd.melt(performance_df, id_vars=['Model'], 
                                value_vars=['Test_R2', 'Train_R2', 'RMSE', 'MAE'],
                                var_name='Metric', value_name='Value')

# 分开绘制R2和误差指标
metrics_r2 = performance_df_melted[performance_df_melted['Metric'].isin(['Test_R2', 'Train_R2'])]
metrics_err = performance_df_melted[performance_df_melted['Metric'].isin(['RMSE', 'MAE'])]

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_r2)
plt.title('Model R² Comparison', fontsize=18)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metrics', loc='upper right')  # 调整图例位置

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_err)
plt.title('Model Error Metrics Comparison', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metrics', loc='upper right')  # 调整图例位置

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')

# 2. 预测vs实际散点图比较 - 分开保存每个模型的散点图
for name, model_results in results.items():
    plt.figure(figsize=(10, 8))  # 单个图的尺寸
    plt.scatter(y_test, model_results["y_pred_test"], alpha=0.7, s=80)
    plt.plot([min(y_test), max(y_test)], 
             [min(y_test), max(y_test)], 'r--', linewidth=2)
    plt.title(f"{name} Model\nR² = {model_results['test_r2']:.4f}", fontsize=20)  # 增大标题字体
    plt.xlabel("Actual C$_{6}$H$_{6}$ Adsorption", fontsize=18)  # 增大X轴标签字体
    plt.ylabel("Predicted C$_{6}$H$_{6}$ Adsorption", fontsize=18)  # 增大Y轴标签字体
    plt.xticks(fontsize=16)  # 增大X轴刻度字体
    plt.yticks(fontsize=16)  # 增大Y轴刻度字体
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存单个模型的散点图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_scatter_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 另外仍然保留一个组合图，以便比较
plt.figure(figsize=(18, 12))
for i, (name, model_results) in enumerate(results.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, model_results["y_pred_test"], alpha=0.7, s=70)
    plt.plot([min(y_test), max(y_test)], 
             [min(y_test), max(y_test)], 'r--', linewidth=2)
    plt.title(f"{name}\nR² = {model_results['test_r2']:.4f}", fontsize=18)  # 统一字体大小
    plt.xlabel("Actual C$_{6}$H$_{6}$ Adsorption", fontsize=16)  # 统一字体大小
    plt.ylabel("Predicted C$_{6}$H$_{6}$ Adsorption", fontsize=16)  # 统一字体大小
    plt.xticks(fontsize=14)  # 增大刻度字体
    plt.yticks(fontsize=14)  # 增大刻度字体
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_scatter_plots_combined.png'), dpi=300, bbox_inches='tight')
plt.close()

# ----- 对最佳模型进行深入分析 -----
print(f"\n对{best_model_name}进行特征重要性和SHAP分析...")

# 1. 内置特征重要性
if hasattr(best_model, 'feature_importances_'):
    best_feature_importance = all_feature_importances[best_model_name]
    
    plt.figure(figsize=(12, 8))
    colors = ['#3498db' if typ=='Continuous' else '#e74c3c' if typ=='Binary' else '#2ecc71' 
              for typ in best_feature_importance['Variable_Type']]
    sns.barplot(x='Importance', y='Feature', data=best_feature_importance, palette=colors)
    plt.title(f'{best_model_name} Feature Importance', fontsize=16)
    plt.xlabel('Relative Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    
    # 添加变量类型图例
    handles = [plt.Rectangle((0,0),1,1,color='#3498db'), 
               plt.Rectangle((0,0),1,1,color='#e74c3c'),
               plt.Rectangle((0,0),1,1,color='#2ecc71')]
    plt.legend(handles, ['Continuous', 'Binary', 'Categorical'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_feature_importance.png'), dpi=300, bbox_inches='tight')

# 2. 排列特征重要性
print("计算排列特征重要性...")
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                       random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': selected_columns,
    'Importance': perm_importance.importances_mean
})
perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
perm_importance_df['Variable_Type'] = perm_importance_df['Feature'].apply(get_variable_type)

perm_importance_df.to_csv(os.path.join(output_dir, f'{best_model_name}_perm_importance.csv'), index=False)

plt.figure(figsize=(12, 8))
colors = ['#3498db' if typ=='Continuous' else '#e74c3c' if typ=='Binary' else '#2ecc71' 
          for typ in perm_importance_df['Variable_Type']]
sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette=colors)
plt.title(f'{best_model_name} Permutation Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# 添加变量类型图例
handles = [plt.Rectangle((0,0),1,1,color='#3498db'), 
           plt.Rectangle((0,0),1,1,color='#e74c3c'),
           plt.Rectangle((0,0),1,1,color='#2ecc71')]
plt.legend(handles, ['Continuous', 'Binary', 'Categorical'], loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{best_model_name}_perm_importance.png'), dpi=300, bbox_inches='tight')

# 3. SHAP值分析
print("执行SHAP值分析...")
# 创建SHAP值计算的目录
shap_dir = os.path.join(output_dir, f"{best_model_name}_shap_analysis")
os.makedirs(shap_dir, exist_ok=True)

try:
    # 为了提高效率，如果数据集很大，可以只使用部分样本来计算SHAP值
    if len(X_test) > 100:
        # 随机选择100个样本
        sampled_indices = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
        X_sample = X_test[sampled_indices]
        y_sample = y_test.iloc[sampled_indices]
    else:
        X_sample = X_test
        y_sample = y_test
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    
    # 计算SHAP值的重要性
    shap_importance = pd.DataFrame({
        'Feature': selected_columns,
        'SHAP_Importance': np.abs(shap_values).mean(axis=0)
    })
    shap_importance = shap_importance.sort_values('SHAP_Importance', ascending=False)
    shap_importance.to_csv(os.path.join(output_dir, f'{best_model_name}_shap_importance.csv'), index=False)
    
    # SHAP摘要图 - 小提琴图
    plt.figure(figsize=(14, 12))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=selected_columns,
        plot_type="violin",
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Summary Plot (Violin): {best_model_name} Model", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_violin_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP摘要图 - 点图
    plt.figure(figsize=(14, 12))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=selected_columns,
        plot_type="dot",
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Summary Plot (Dot): {best_model_name} Model", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_dot_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP条形图
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=selected_columns,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Feature Importance: {best_model_name} Model", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 为前5个重要特征创建SHAP依赖图
    top_features = shap_importance['Feature'].iloc[:5].tolist()
    for feature in top_features:
        plt.figure(figsize=(10, 8))
        feature_idx = selected_columns.index(feature)
        shap.dependence_plot(
            feature_idx, 
            shap_values, 
            X_sample,
            feature_names=selected_columns,
            show=False
        )
        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=20)
        plt.xlabel(feature, fontsize=18)
        plt.ylabel("SHAP Value", fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"shap_dependence_{feature}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # SHAP力图 - 为前5个样本分别创建
    X_sample_df = pd.DataFrame(X_sample, columns=selected_columns)
    
    for i in range(min(5, len(X_sample))):
        plt.figure(figsize=(20, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X_sample_df.iloc[i],
            feature_names=selected_columns,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot for Sample {i+1}: {best_model_name} Model", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"shap_force_plot_sample_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
except Exception as e:
    print(f"SHAP分析过程中出现错误: {e}")

# 创建综合特征重要性比较
print("创建综合特征排名...")

# 收集所有特征重要性方法的结果
combined_rankings = pd.DataFrame({'Feature': selected_columns})

# 添加内置特征重要性排名
if hasattr(best_model, 'feature_importances_'):
    model_importance = pd.Series(best_model.feature_importances_, index=selected_columns)
    combined_rankings['Built_in_Rank'] = model_importance.rank(ascending=False)

# 添加排列特征重要性排名
perm_importance_series = pd.Series(perm_importance.importances_mean, index=selected_columns)
combined_rankings['Perm_Rank'] = perm_importance_series.rank(ascending=False)

# 添加SHAP特征重要性排名
shap_importance_series = pd.Series(shap_importance['SHAP_Importance'].values, index=shap_importance['Feature'])
combined_rankings['SHAP_Rank'] = shap_importance_series.rank(ascending=False)

# 计算平均排名
rank_columns = [col for col in combined_rankings.columns if col.endswith('_Rank')]
combined_rankings['Mean_Rank'] = combined_rankings[rank_columns].mean(axis=1)

# 排序并添加变量类型
combined_rankings = combined_rankings.sort_values('Mean_Rank')
combined_rankings['Variable_Type'] = combined_rankings['Feature'].apply(get_variable_type)

# 保存综合排名
combined_rankings.to_csv(os.path.join(output_dir, f'{best_model_name}_combined_ranking.csv'), index=False)

# 绘制综合排名条形图
plt.figure(figsize=(12, 10))
top_features = combined_rankings.head(10)
colors = ['#3498db' if typ=='Continuous' else '#e74c3c' if typ=='Binary' else '#2ecc71' 
          for typ in top_features['Variable_Type']]
sns.barplot(x='Mean_Rank', y='Feature', data=top_features, palette=colors)
plt.title('Combined Feature Importance Ranking (Top 10)', fontsize=18)
plt.xlabel('Mean Rank (Lower = More Important)', fontsize=16)
plt.ylabel('Feature', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 添加变量类型图例
handles = [plt.Rectangle((0,0),1,1,color='#3498db'), 
           plt.Rectangle((0,0),1,1,color='#e74c3c'),
           plt.Rectangle((0,0),1,1,color='#2ecc71')]
plt.legend(handles, ['Continuous', 'Binary', 'Categorical'], loc='lower right', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{best_model_name}_combined_ranking_barplot.png'), dpi=300)

# 创建特征重要性综合报告
with open(os.path.join(output_dir, f'{best_model_name}_feature_importance_report.md'), 'w', encoding='utf-8') as f:
    f.write(f"# C$_{6}$H$_{6}$ Adsorption Feature Importance Analysis Report - {best_model_name} Model\n\n")
    
    f.write("## Dataset Information\n")
    f.write(f"- Records: {len(df_cleaned)}\n")
    f.write(f"- Features: {len(selected_columns)}\n\n")
    
    f.write("## Model Performance Comparison\n")
    f.write("| Model | Train R² | Test R² | RMSE | MAE |\n")
    f.write("|------|---------:|--------:|-----:|----:|\n")
    
    for _, row in performance_df.iterrows():
        f.write(f"| {row['Model']} | {row['Train_R2']:.4f} | {row['Test_R2']:.4f} | ")
        f.write(f"{row['RMSE']:.4f} | {row['MAE']:.4f} |\n")
    
    f.write(f"\n## Best Model: {best_model_name}\n")
    f.write(f"- Test R²: {results[best_model_name]['test_r2']:.4f}\n")
    f.write(f"- Test RMSE: {results[best_model_name]['test_rmse']:.4f}\n\n")
    
    f.write("## Best Model Parameters\n")
    f.write("```\n")
    for param, value in best_params[best_model_name].items():
        f.write(f"{param}: {value}\n")
    f.write("```\n\n")
    
    f.write("## Combined Feature Importance Ranking (Top 10)\n")
    f.write("| Rank | Feature | Type | Mean Rank | Built-in Rank | Perm Rank | SHAP Rank |\n")
    f.write("|-----:|------|----------|--------:|--------:|--------:|--------:|\n")
    
    for i, (_, row) in enumerate(combined_rankings.head(10).iterrows()):
        f.write(f"| {i+1} | {row['Feature']} | {row['Variable_Type']} | {row['Mean_Rank']:.2f} | ")
        if 'Built_in_Rank' in row:
            f.write(f"{row['Built_in_Rank']:.0f} | ")
        else:
            f.write("N/A | ")
        f.write(f"{row['Perm_Rank']:.0f} | {row['SHAP_Rank']:.0f} |\n")
        
    f.write("\n## Analysis Conclusion\n")
    f.write("Based on multiple feature importance evaluation methods (built-in, permutation importance, SHAP values), the following features have the most significant impact on C$_{6}$H$_{6}$ adsorption performance:\n\n")
    
    for i, (_, row) in enumerate(combined_rankings.head(5).iterrows()):
        f.write(f"{i+1}. **{row['Feature']}**: {row['Variable_Type']}, Mean Rank {row['Mean_Rank']:.2f}\n")
    
    f.write("\nThese features should be the focus when optimizing materials for C$_{6}$H$_{6}$ adsorption.\n")

print(f"\n分析完成。所有结果已保存至：{output_dir}")
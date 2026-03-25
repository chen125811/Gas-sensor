# -*- coding: utf-8 -*-
# lightgbm_shap_importance_analysis.py - 使用LightGBM模型的SHAP分析计算变量重要性占比

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
import shap
import warnings

# 忽略FutureWarning警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置输出目录
output_dir = "F:\\VS code\\code for gas adsorption\\NH3\\各种树模型\\lightgbm_shap_analysis"
os.makedirs(output_dir, exist_ok=True)

# 设置文件路径
data_path = "F:\\VS code\\code for gas adsorption\\gas adsorption-NH3.xlsx"
model_params_path = "F:\\VS code\\code for gas adsorption\\NH3\\各种树模型\\tree_models_results\\best_model_parameters.json"

# 设置图表样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

print("开始使用LightGBM进行SHAP重要性分析...")

try:
    # 1. 加载最佳模型参数
    print("加载最佳模型参数...")
    with open(model_params_path, 'r') as f:
        best_params = json.load(f)
    
    # 获取LightGBM参数
    if "LightGBM" in best_params:
        model_params = best_params["LightGBM"]
        print(f"使用LightGBM参数: {model_params}")
    else:
        raise ValueError("没有找到LightGBM参数")
    
    # 2. 准备数据集
    print("准备数据集...")
    df = pd.read_excel(data_path)
    
    # 重命名列
    df = df.rename(columns={
        "BET Specific surface area": "BET_surface_area", 
        "Pore volume": "Pore_volume",
        "Other active sites": "Other_active_sites", 
        "Open metal sites": "Open_metal_sites",
        "Hydroxyl": "Hydroxyl", 
        "Amino": "Amino", 
        "carboxyl": "Carboxyl",
        "Gas concentration": "Gas_concentration", 
        "Measurement method": "Measurement_method",
        "Metal Node": "Metal_Node", 
        "Filling gas": "Filling_gas",
        "Synthesis temperature(℃)": "Synthesis_temperature", 
        "Synthesis time (h)": "Synthesis_time",
        "Adsorption level vs BPP-5": "Adsorption_Level"
    })
    
    # 移除字符串中的星号
    def remove_asterisk(value):
        if isinstance(value, str):
            return value.replace("*", "")
        return value
    
    # 数据清洗
    selected_columns = [
        "BET_surface_area", "Pore_volume", "Other_active_sites", "Open_metal_sites", 
        "Hydroxyl", "Amino", "Carboxyl", "Gas_concentration", "Measurement_method",
        "Metal_Node", "Filling_gas", "Synthesis_temperature", "Synthesis_time"
    ]
    
    df[selected_columns] = df[selected_columns].applymap(remove_asterisk)
    
    # 处理分类变量
    categorical_columns = ["Other_active_sites", "Metal_Node", 
                           "Measurement_method", "Filling_gas"]
    for col in categorical_columns:
        df[col] = df[col].astype(str).factorize()[0]
    
    # 处理二值变量
    binary_columns = ["Hydroxyl", "Amino", "Carboxyl", "Open_metal_sites"]
    for col in binary_columns:
        df[col] = df[col].replace({'0': 0, '1': 1, 0.0: 0, 1.0: 1})
    
    # 转换为浮点数
    df[selected_columns] = df[selected_columns].astype(float)
    
    # 去除缺失值
    df_cleaned = df.dropna(subset=selected_columns + ["Adsorption_Level"]).copy()
    print(f"数据集大小: {len(df_cleaned)} 条记录")
    
    # 3. 数据准备
    X = df_cleaned[selected_columns]
    y = df_cleaned["Adsorption_Level"]
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_columns)
    
    # 4. 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # 5. 转换参数类型 (JSON会将数字存为字符串)
    params_converted = {}
    for key, value in model_params.items():
        if isinstance(value, str):
            try:
                # 尝试将字符串转换为数字
                if '.' in value:
                    params_converted[key] = float(value)
                else:
                    params_converted[key] = int(value)
            except ValueError:
                # 如果无法转换为数字，保留字符串
                params_converted[key] = value
        else:
            params_converted[key] = value
    
    # 确保有random_state参数
    if 'random_state' not in params_converted:
        params_converted['random_state'] = 42
    
    # 6. 创建并训练LightGBM模型
    print("训练LightGBM模型...")
    model = lgb.LGBMRegressor(**params_converted)
    model.fit(X_train, y_train)
    
    # 7. 评估模型性能
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"LightGBM模型性能 - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # 8. 计算SHAP值
    print("\n计算SHAP值...")
    explainer = shap.TreeExplainer(model)

    np.random.seed(42)  # 固定种子与多种树模型2-NH3.py一致
    if len(X_test) > 100:
        # 从X_test直接选择样本
        sample_indices = np.random.choice(len(X_test), 100, replace=False)
        X_sample = X_test[sample_indices]
    else:
        X_sample = X_test

    # 确保使用DataFrame而不是numpy数组计算SHAP值
    shap_values = explainer(X_sample)

    # 9. 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values.values).mean(0)
    
    # 10. 计算每个特征的SHAP重要性占比
    total_shap = mean_abs_shap.sum()
    shap_importance_df = pd.DataFrame({
        'Feature': selected_columns,
        'Mean_Abs_SHAP': mean_abs_shap,
        'SHAP_Importance_Pct': mean_abs_shap / total_shap * 100
    }).sort_values('SHAP_Importance_Pct', ascending=False)
    
    # 输出每个特征的SHAP重要性占比
    print("\nSHAP重要性占比分析结果:")
    for _, row in shap_importance_df.iterrows():
        print(f"{row['Feature']}: {row['SHAP_Importance_Pct']:.2f}%")
    
    # 11. 可视化SHAP重要性占比
    plt.figure(figsize=(12, 8))
    # 高亮显示前三个最重要的特征
    top3_features = shap_importance_df.head(3)['Feature'].tolist()
    colors = ['#d62728' if feat in top3_features else '#1f77b4' 
              for feat in shap_importance_df['Feature']]
    sns.barplot(x='SHAP_Importance_Pct', y='Feature', hue='Feature', 
               data=shap_importance_df, palette=dict(zip(shap_importance_df['Feature'], colors)), legend=False)
    plt.title('Feature Importance by SHAP Values (%)', fontsize=22)
    plt.xlabel('SHAP Importance (%)', fontsize=18)
    plt.ylabel('Feature', fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_importance_percentage.png"), dpi=1200, bbox_inches='tight')
    
    # 12. SHAP摘要图 - 更详细的分析
    plt.figure(figsize=(14, 10))
    # 确保使用feature_names参数传递特征名称
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, 
                     feature_names=selected_columns)  # 明确指定特征名称
    plt.title('SHAP Feature Importance Summary', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=1200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 10))
    # 同样为点图指定特征名称
    shap.summary_plot(shap_values, X_sample, show=False, 
                     feature_names=selected_columns)  # 明确指定特征名称
    plt.title('SHAP Summary Plot', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_dot.png"), dpi=1200, bbox_inches='tight')
    plt.close()
    
    # 13. 保存结果到CSV
    shap_importance_df.to_csv(os.path.join(output_dir, "shap_importance_percentage.csv"), index=False)
    
    # 14. 生成综合报告
    with open(os.path.join(output_dir, "shap_importance_report.md"), 'w', encoding='utf-8') as f:
        f.write("# LightGBM模型SHAP重要性分析报告\n\n")
        
        f.write("## 模型性能\n")
        f.write(f"- R²: {r2:.4f}\n")
        f.write(f"- RMSE: {rmse:.4f}\n\n")
        
        f.write("## 特征SHAP重要性占比\n")
        f.write("| 特征 | SHAP重要性占比 (%) | 累积占比 (%) |\n")
        f.write("|------|----------------:|-------------:|\n")
        
        cumulative = 0
        for _, row in shap_importance_df.iterrows():
            cumulative += row['SHAP_Importance_Pct']
            f.write(f"| {row['Feature']} | {row['SHAP_Importance_Pct']:.2f} | {cumulative:.2f} |\n")
        
        f.write("## 结论\n\n")
        
        # 确定最重要的3个特征
        top_features = shap_importance_df.head(3)['Feature'].tolist()
        # 修改这里，将C6H6改为NH3
        f.write("基于SHAP值分析，以下特征对NH3吸附贡献最大：\n\n")
        
        for i, feature in enumerate(top_features):
            importance = shap_importance_df[shap_importance_df['Feature'] == feature]['SHAP_Importance_Pct'].values[0]
            f.write(f"{i+1}. **{feature}**: {importance:.2f}%\n")
        
        top3_importance_sum = sum(shap_importance_df.head(3)['SHAP_Importance_Pct'])
        f.write(f"\n前三大特征共同贡献了模型预测能力的 **{top3_importance_sum:.2f}%**，")
        if top3_importance_sum > 50:
            f.write("表明它们是决定NH3吸附性能的关键因素。\n")
        else:
            f.write("表明它们是重要因素，但其他变量也有显著贡献。\n")
    
    #15. 计算SHAP值的统计显著性
    def calculate_shap_significance(model, X, feature_names, n_permutations=100, random_state=42):
        """使用排列检验法计算SHAP值的统计显著性"""
        print("计算SHAP值的统计显著性...")
        
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        # 计算原始SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        original_mean_abs_shap = np.abs(shap_values.values).mean(0)
        
        # 为每个特征计算排列显著性
        p_values = []
        np.random.seed(random_state)
        
        for i, feature in enumerate(feature_names):
            print(f"  处理特征 {i+1}/{len(feature_names)}: {feature}")
            # 存储排列后的SHAP值
            permuted_shap_means = []
            
            # 创建X的副本以进行排列
            X_permuted = X.copy()
            
            for p in range(n_permutations):
                if p % 20 == 0 and p > 0:
                    print(f"    完成排列 {p}/{n_permutations}")
                    
                # 随机打乱特定特征的值
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                # 计算排列后的SHAP值
                permuted_shap = explainer(X_permuted)
                mean_abs_permuted_shap = np.abs(permuted_shap.values).mean(0)
                
                # 存储该特征的排列后的平均绝对SHAP值
                permuted_shap_means.append(mean_abs_permuted_shap[i])
            
            # 计算p值：(排列后SHAP值 >= 原始SHAP值)的比例
            p_value = np.mean(np.array(permuted_shap_means) >= original_mean_abs_shap[i])
            p_values.append(p_value)
        
        # 创建包含p值的DataFrame
        significance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': original_mean_abs_shap,
            'P_Value': p_values,
            'Significant': np.array(p_values) < 0.05  # 使用0.05作为显著性水平
        })
        
        return significance_df.sort_values('Mean_Abs_SHAP', ascending=False)
    
    # 计算特征SHAP值的统计显著性
    # shap_significance_df = calculate_shap_significance(model, X_scaled_df, selected_columns)
    
    # 仅保留这一个显著性计算:
    print("\n准备计算特征重要性的统计显著性...")
    # 如果数据集很大，可以使用较小的样本来加速计算
    if len(X_sample) > 50:
        # 确保样本是DataFrame
        if isinstance(X_sample, pd.DataFrame):
            significance_sample = X_sample.iloc[:50]
        else:
            significance_sample = pd.DataFrame(X_sample[:50], columns=selected_columns)
    else:
        if not isinstance(X_sample, pd.DataFrame):
            significance_sample = pd.DataFrame(X_sample, columns=selected_columns)
        else:
            significance_sample = X_sample
    
    # 16. 保存SHAP值显著性分析结果
    # significance_df.to_csv(os.path.join(output_dir, "shap_significance_analysis.csv"), index=False)
    
    # 计算显著性水平 (可选，因为这个过程可能较慢)
    try:
        print("\n计算特征重要性的统计显著性...")
        
        # 计算显著性
        significance_df = calculate_shap_significance(
            model, significance_sample, selected_columns, n_permutations=50)
        
        print("\n特征重要性显著性分析结果:")
        for _, row in significance_df.iterrows():
            significance = "显著" if row['Significant'] else "不显著"
            print(f"{row['Feature']}: p值 = {row['P_Value']:.4f} ({significance})")
        
        # 保存显著性结果
        significance_df.to_csv(os.path.join(output_dir, "shap_significance.csv"), index=False)
        
        # 可视化显著性结果
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Mean_Abs_SHAP', 
            y='Feature', 
            data=significance_df,
            palette=['#2ecc71' if sig else '#95a5a6' for sig in significance_df['Significant']]
        )
        plt.title('Feature Importance with Significance', fontsize=22)
        plt.xlabel('Mean |SHAP| Value', fontsize=18)
        plt.ylabel('Feature', fontsize=18)
        
        # 添加p值标注
        for i, row in enumerate(significance_df.itertuples()):
            plt.text(
                row.Mean_Abs_SHAP + 0.01, 
                i, 
                f"p={row.P_Value:.3f}{'*' if row.Significant else ''}", 
                va='center'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_significance.png"), dpi=1200, bbox_inches='tight')
        plt.close()
        
        # 在报告中添加显著性信息
        with open(os.path.join(output_dir, "shap_significance_report.md"), 'w', encoding='utf-8') as f:
            f.write("# SHAP重要性统计显著性分析\n\n")
            
            f.write("## 特征重要性显著性\n")
            f.write("| 特征 | SHAP值 | p值 | 显著性 |\n")
            f.write("|------|-------|-----|--------|\n")
            
            for _, row in significance_df.iterrows():
                significance = "显著 ✓" if row['Significant'] else "不显著 ✗"
                f.write(f"| {row['Feature']} | {row['Mean_Abs_SHAP']:.4f} | {row['P_Value']:.4f} | {significance} |\n")
            
            f.write("\n## BET比表面积和孔容的显著性\n")
            bet_sig = significance_df[significance_df['Feature'] == 'BET_surface_area']
            pore_sig = significance_df[significance_df['Feature'] == 'Pore_volume']
            
            if not bet_sig.empty:
                bet_significant = bet_sig['Significant'].values[0]
                f.write(f"- BET比表面积: p值 = {bet_sig['P_Value'].values[0]:.4f} ")
                f.write("(统计学显著)\n" if bet_significant else "(统计学不显著)\n")
                
            if not pore_sig.empty:
                pore_significant = pore_sig['Significant'].values[0]
                f.write(f"- 孔容: p值 = {pore_sig['P_Value'].values[0]:.4f} ")
                f.write("(统计学显著)\n" if pore_significant else "(统计学不显著)\n")
            
            f.write("\n## 结论\n\n")
            significant_features = significance_df[significance_df['Significant']]['Feature'].tolist()
            
            # 将以下内容中的C6H6改为NH3
            if len(significant_features) > 0:
                f.write(f"在显著性水平α=0.05下，以下{len(significant_features)}个特征对NH3吸附的影响具有统计学显著性：\n\n")
                for i, feature in enumerate(significant_features):
                    f.write(f"{i+1}. **{feature}**\n")
            else:
                f.write("在显著性水平α=0.05下，没有特征表现出统计学显著性。这可能是由于样本量不足或模型不稳定导致的。\n")
        
    except Exception as e:
        print(f"计算统计显著性时出错: {e}")
        print("继续执行其余分析...")
    
    print(f"\n分析完成。所有结果已保存至: {output_dir}")
    
except Exception as e:
    print(f"分析过程中出错: {e}")
    import traceback
    traceback.print_exc()


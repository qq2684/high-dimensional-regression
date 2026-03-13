"""
对比实验模块
在不同高维设置下对比LASSO, Elastic Net, SCAD的性能
"""

import numpy as np
import pandas as pd
import warnings
from data_generation import generate_data, split_data
from methods import LassoRegression, ElasticNetRegression, SCADRegression

warnings.filterwarnings('ignore')


def compute_metrics(y_true, y_pred, beta_true, beta_pred, signal_idx, tol=1e-3):
    """
    计算性能指标
    
    返回：
    ------
    metrics : dict
        包含各项指标的字典
    """
    # 预测误差
    test_mse = np.mean((y_true - y_pred) ** 2)
    
    # 参数估计误差
    beta_error = np.linalg.norm(beta_pred - beta_true, ord=2)
    
    # 变量选择性能
    pred_idx = np.where(np.abs(beta_pred) > tol)[0]
    
    # True Positive Rate (TPR): 正确选中真实变量的比率
    if len(signal_idx) > 0:
        tp = len(np.intersect1d(pred_idx, signal_idx))
        tpr = tp / len(signal_idx)
    else:
        tp = 0
        tpr = 0
    
    # False Discovery Rate (FDR): 错误选中变量的比率
    if len(pred_idx) > 0:
        fp = len(pred_idx) - tp
        fdr = fp / len(pred_idx)
    else:
        fdr = 0
    
    # 选中变量个数
    num_selected = len(pred_idx)
    
    return {
        'test_mse': test_mse,
        'beta_error': beta_error,
        'tpr': tpr,
        'fdr': fdr,
        'num_selected': num_selected,
        'tp': tp
    }


def run_single_experiment(n, p, s, method_name, seed=42):
    """
    运行单次实验
    """
    # 生成数据
    X, y, beta_true, signal_idx = generate_data(n, p, s, snr=3.0, seed=seed)
    X_train, X_test, y_train, y_test, _, _ = split_data(X, y, test_size=0.2, seed=seed)
    
    # 选择方法
    if method_name == 'LASSO':
        model = LassoRegression()
    elif method_name == 'Elastic Net':
        model = ElasticNetRegression()
    elif method_name == 'SCAD':
        model = SCADRegression()
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    try:
        # 训练模型
        model.fit(X_train, y_train, cv_folds=5)
        
        # 计算指标
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, beta_true, model.beta, signal_idx)
        metrics['method'] = method_name
        metrics['n'] = n
        metrics['p'] = p
        metrics['s'] = s
        metrics['lambda'] = model.lambda_opt
        
        return metrics
    except Exception as e:
        print(f"Error in {method_name} (n={n}, p={p}): {str(e)}")
        return None


def run_comprehensive_experiment():
    """
    运行完整的对比实验
    """
    # 实验参数设置
    n_list = [100, 200, 500]
    p_list = [500, 1000, 2000]
    s = 10  # 稀疏度固定为10
    methods = ['LASSO', 'Elastic Net', 'SCAD']
    num_replicates = 10  # 每个设置重复10次
    
    results = []
    
    total_experiments = len(n_list) * len(p_list) * len(methods) * num_replicates
    experiment_count = 0
    
    print("开始运行对比实验...")
    print(f"总共需要运行 {total_experiments} 个实验")
    print("-" * 60)
    
    for n in n_list:
        for p in p_list:
            for method in methods:
                for rep in range(num_replicates):
                    experiment_count += 1
                    
                    # 运行实验
                    result = run_single_experiment(n, p, s, method, seed=42+rep)
                    
                    if result is not None:
                        results.append(result)
                    
                    if experiment_count % 30 == 0:
                        print(f"已完成 {experiment_count}/{total_experiments} 个实验")
    
    print("-" * 60)
    print("实验完成！")
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def summarize_results(results_df):
    """
    汇总实验结果
    """
    summary = results_df.groupby(['n', 'p', 's', 'method']).agg({
        'test_mse': ['mean', 'std'],
        'beta_error': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'num_selected': ['mean', 'std'],
    }).round(4)
    
    return summary


if __name__ == '__main__':
    # 运行实验
    results_df = run_comprehensive_experiment()
    
    # 保存结果
    results_df.to_csv('../results/experiment_results.csv', index=False)
    print("\n结果已保存到 results/experiment_results.csv")
    
    # 打印汇总
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    summary = summarize_results(results_df)
    print(summary)

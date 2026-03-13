"""
数据生成模块
生成高维线性回归模拟数据
"""

import numpy as np
from scipy.linalg import toeplitz


def generate_data(n, p, s, snr=3.0, rho=0.5, seed=42):
    """
    生成高维线性回归模拟数据
    
    参数：
    ------
    n : int
        样本量
    p : int
        特征维度
    s : int
        真实非零系数个数（稀疏度）
    snr : float
        信噪比 (signal-to-noise ratio)
    rho : float
        特征相关系数（Toeplitz结构）
    seed : int
        随机种子
        
    返回：
    ------
    X : array, shape (n, p)
        设计矩阵
    y : array, shape (n,)
        响应变量
    beta_true : array, shape (p,)
        真实系数
    """
    
    np.random.seed(seed)
    
    # 生成具有结构相关性的特征矩阵
    # 使用Toeplitz协方差矩阵结构
    cov_matrix = toeplitz(rho ** np.arange(p))
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    # Cholesky分解生成相关特征
    L = np.linalg.cholesky(cov_matrix)
    X_std = np.random.normal(0, 1, size=(n, p))
    X = X_std @ L.T
    
    # 标准化特征
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # 生成真实系数（稀疏）
    beta_true = np.zeros(p)
    signal_idx = np.random.choice(p, size=s, replace=False)
    beta_true[signal_idx] = np.random.uniform(2, 5, size=s) * np.sign(np.random.randn(s))
    
    # 生成响应变量
    y_signal = X @ beta_true
    signal_var = np.var(y_signal)
    noise_var = signal_var / snr
    noise = np.random.normal(0, np.sqrt(noise_var), size=n)
    y = y_signal + noise
    
    return X, y, beta_true, signal_idx


def split_data(X, y, test_size=0.2, seed=42):
    """
    将数据分为训练集和测试集
    """
    np.random.seed(seed)
    n = X.shape[0]
    test_idx = np.random.choice(n, size=int(n * test_size), replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test, train_idx, test_idx

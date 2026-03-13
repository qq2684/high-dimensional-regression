"""
高维回归方法实现模块
包括 LASSO, Elastic Net, 和 SCAD
"""

import numpy as np
from scipy.optimize import minimize, fminbound
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler


class SparseRegression:
    """高维回归基类"""
    
    def __init__(self):
        self.beta = None
        self.lambda_opt = None
        
    def fit(self, X, y):
        raise NotImplementedError
        
    def predict(self, X):
        return X @ self.beta
    
    def get_selected_features(self, tol=1e-3):
        """获取选中的特征索引"""
        return np.where(np.abs(self.beta) > tol)[0]


class LassoRegression(SparseRegression):
    """LASSO 回归"""
    
    def fit(self, X, y, lambda_seq=None, cv_folds=5):
        """
        使用交叉验证选择最优lambda
        """
        if lambda_seq is None:
            # 生成lambda序列
            lambda_max = np.max(np.abs(X.T @ y)) / len(y)
            lambda_seq = np.logspace(np.log10(lambda_max * 0.001), 
                                     np.log10(lambda_max), 100)
        
        best_mse = np.inf
        best_lambda = None
        
        for lam in lambda_seq:
            lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
            
            # 简单的交叉验证
            n = X.shape[0]
            fold_size = n // cv_folds
            cv_mse = []
            
            for fold in range(cv_folds):
                val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_idx = np.setdiff1d(np.arange(n), val_idx)
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                lasso.fit(X_train, y_train)
                y_pred = lasso.predict(X_val)
                mse = np.mean((y_pred - y_val) ** 2)
                cv_mse.append(mse)
            
            mean_mse = np.mean(cv_mse)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_lambda = lam
        
        # 用最优lambda在全数据上训练
        lasso = Lasso(alpha=best_lambda, fit_intercept=False, max_iter=10000)
        lasso.fit(X, y)
        self.beta = lasso.coef_
        self.lambda_opt = best_lambda
        return self


class ElasticNetRegression(SparseRegression):
    """Elastic Net 回归"""
    
    def fit(self, X, y, lambda_seq=None, l1_ratio=0.5, cv_folds=5):
        """
        使用交叉验证选择最优lambda
        l1_ratio: L1正则化的权重比例 (0-1)
        """
        if lambda_seq is None:
            lambda_max = np.max(np.abs(X.T @ y)) / len(y)
            lambda_seq = np.logspace(np.log10(lambda_max * 0.001), 
                                     np.log10(lambda_max), 100)
        
        best_mse = np.inf
        best_lambda = None
        
        for lam in lambda_seq:
            en = ElasticNet(alpha=lam, l1_ratio=l1_ratio, 
                           fit_intercept=False, max_iter=10000)
            
            n = X.shape[0]
            fold_size = n // cv_folds
            cv_mse = []
            
            for fold in range(cv_folds):
                val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_idx = np.setdiff1d(np.arange(n), val_idx)
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                en.fit(X_train, y_train)
                y_pred = en.predict(X_val)
                mse = np.mean((y_pred - y_val) ** 2)
                cv_mse.append(mse)
            
            mean_mse = np.mean(cv_mse)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_lambda = lam
        
        en = ElasticNet(alpha=best_lambda, l1_ratio=l1_ratio, 
                       fit_intercept=False, max_iter=10000)
        en.fit(X, y)
        self.beta = en.coef_
        self.lambda_opt = best_lambda
        return self


class SCADRegression(SparseRegression):
    """SCAD (Smoothly Clipped Absolute Deviation) 回归"""
    
    @staticmethod
    def scad_penalty(x, lam, a=3.7):
        """SCAD惩罚函数"""
        abs_x = np.abs(x)
        
        if abs_x <= lam:
            return lam * abs_x
        elif abs_x <= a * lam:
            return -((abs_x**2 - 2*a*lam*abs_x + lam**2) / (2*(a-1)))
        else:
            return (a+1)*lam**2 / 2
    
    def scad_derivative(self, x, lam, a=3.7):
        """SCAD惩罚函数的导数（近似）"""
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        
        if abs_x <= lam:
            return lam * sign_x
        elif abs_x <= a * lam:
            return -abs_x / (a - 1) + a * lam * sign_x / (a - 1)
        else:
            return 0
    
    def fit(self, X, y, lambda_seq=None, cv_folds=5, a=3.7):
        """
        使用local quadratic approximation和交叉验证
        """
        if lambda_seq is None:
            lambda_max = np.max(np.abs(X.T @ y)) / len(y)
            lambda_seq = np.logspace(np.log10(lambda_max * 0.001), 
                                     np.log10(lambda_max), 50)
        
        best_mse = np.inf
        best_lambda = None
        
        for lam in lambda_seq:
            n = X.shape[0]
            fold_size = n // cv_folds
            cv_mse = []
            
            for fold in range(cv_folds):
                val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_idx = np.setdiff1d(np.arange(n), val_idx)
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 初始化系数
                beta = np.zeros(X_train.shape[1])
                
                # 通过local quadratic approximation迭代优化
                for it in range(100):
                    # 计算梯度
                    residuals = y_train - X_train @ beta
                    grad = -X_train.T @ residuals / len(y_train)
                    
                    # 添加SCAD惩罚梯度
                    for j in range(len(beta)):
                        grad[j] += self.scad_derivative(beta[j], lam, a) / len(y_train)
                    
                    # 简单的梯度下降
                    step_size = 0.01
                    beta = beta - step_size * grad
                
                y_pred = X_val @ beta
                mse = np.mean((y_pred - y_val) ** 2)
                cv_mse.append(mse)
            
            mean_mse = np.mean(cv_mse)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_lambda = lam
        
        # 用最优lambda重新训练
        self.beta = np.zeros(X.shape[1])
        for it in range(100):
            residuals = y - X @ self.beta
            grad = -X.T @ residuals / len(y)
            for j in range(len(self.beta)):
                grad[j] += self.scad_derivative(self.beta[j], best_lambda, a) / len(y)
            step_size = 0.01
            self.beta = self.beta - step_size * grad
        
        self.lambda_opt = best_lambda
        return self

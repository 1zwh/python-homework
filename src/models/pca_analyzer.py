import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings

class PCAAnalyzer:
    """主成分分析器"""
    
    def __init__(self, n_components: Optional[int] = None, 
                robust: bool = False,
                kernel: Optional[str] = None,
                random_state: int = 42):
        
        self.n_components = n_components
        self.robust = robust
        self.kernel = kernel
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.loadings_ = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> 'PCAAnalyzer':
        """拟合PCA模型"""
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
    
        # 添加调试信息
        print(f"输入数据形状: {X.shape}")
        print(f"特征数量: {len(self.feature_names)}")
    
     # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
    
        # 选择PCA类型
        if self.kernel:
            self.pca = KernelPCA(n_components=self.n_components,
                            kernel=self.kernel,
                            fit_inverse_transform=True,
                            random_state=self.random_state)
        elif self.robust:
            self.pca = self._create_robust_pca(X_scaled)
        else:
            self.pca = PCA(n_components=self.n_components,
                        random_state=self.random_state)
    
    # 拟合PCA
        self.pca.fit(X_scaled)
    
    # 保存结果
        if hasattr(self.pca, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
            self.eigenvalues_ = self.pca.explained_variance_
            print(f"特征值: {self.eigenvalues_.shape}")
            print(f"特征值: {self.eigenvalues_}")
    
        if hasattr(self.pca, 'components_'):
            self.components_ = self.pca.components_
            print(f"成分矩阵形状: {self.components_.shape}")
            self._compute_loadings(X_scaled)
    
        return self
    
    def _create_robust_pca(self, X_scaled: np.ndarray):
        """创建稳健PCA"""
        class RobustPCA:
            def __init__(self, n_components=None):
                self.n_components = n_components
                
            def fit(self, X):
                # 使用MCD稳健协方差估计
                mcd = MinCovDet(random_state=42)
                mcd.fit(X)
                cov_matrix = mcd.covariance_
                
                # 特征值分解
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # 排序
                idx = eigenvalues.argsort()[::-1]
                self.eigenvalues_ = eigenvalues[idx]
                self.components_ = eigenvectors[:, idx]
                
                if self.n_components is not None:
                    self.components_ = self.components_[:, :self.n_components]
                    self.eigenvalues_ = self.eigenvalues_[:self.n_components]
                
                self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(self.eigenvalues_)
                return self
            
            def transform(self, X):
                return X @ self.components_
        
        return RobustPCA(n_components=self.n_components)
    
    def _compute_loadings(self, X_scaled: np.ndarray):
        """计算特征载荷"""
        if self.components_ is not None and self.eigenvalues_ is not None:
            # 计算特征值的平方根
            sqrt_eigenvalues = np.sqrt(self.eigenvalues_)
            
            # components_ 形状: (n_components, n_features)
            # sqrt_eigenvalues 形状: (n_components,)
            
            # 正确的广播乘法
            self.loadings_ = self.components_ * sqrt_eigenvalues.reshape(-1, 1)
            
            # 确保形状正确
            n_components, n_features = self.components_.shape
            assert self.loadings_.shape == (n_components, n_features)
            
            # 创建DataFrame：每行是一个主成分，每列是一个特征
            self.loadings_df = pd.DataFrame(
                self.loadings_,  # 形状: (n_components, n_features)
                columns=self.feature_names,
                index=[f'PC{i+1}' for i in range(n_components)]
            )
    
    def determine_optimal_components(self, X: np.ndarray) -> Dict[str, int]:
        """确定最优主成分数量"""
        results = {}
        
        # 特征值>1准则
        eigenvalues = self.eigenvalues_
        n_eigenvalue = np.sum(eigenvalues > 1)
        results['eigenvalue'] = n_eigenvalue
        
        # 累计方差≥85%
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        n_variance = np.argmax(cumulative_variance >= 0.85) + 1
        if n_variance == 1 and cumulative_variance[0] < 0.85:
            n_variance = len(cumulative_variance)
        results['variance'] = n_variance
        
        # 碎石图拐点
        n_scree = self._scree_test(eigenvalues)
        results['scree'] = n_scree
        
        # 平行分析
        n_parallel = self._parallel_analysis(X)
        results['parallel'] = n_parallel
        
        return results
    
    def _scree_test(self, eigenvalues: np.ndarray) -> int:
        """碎石图拐点测试"""
        # 计算二阶差分
        diffs = np.diff(eigenvalues)
        diffs2 = np.diff(diffs)
        
        if len(diffs2) > 0:
            elbow = np.argmax(diffs2) + 2
        else:
            elbow = np.sum(eigenvalues > 1)
        
        return max(1, min(elbow, len(eigenvalues)))
    
    def _parallel_analysis(self, X: np.ndarray, n_permutations: int = 100) -> int:
        """平行分析"""
        n_samples, n_features = X.shape
        
        random_eigenvalues = []
        for _ in range(n_permutations):
            # 生成随机数据
            X_random = np.zeros_like(X)
            for j in range(n_features):
                X_random[:, j] = np.random.permutation(X[:, j])
            
            # 计算特征值
            cov_random = np.cov(X_random, rowvar=False)
            eigvals_random = np.linalg.eigvalsh(cov_random)
            random_eigenvalues.append(np.sort(eigvals_random)[::-1])
        
        random_eigenvalues = np.array(random_eigenvalues)
        threshold = np.percentile(random_eigenvalues, 95, axis=0)
        
        n_components = np.sum(self.eigenvalues_ > threshold)
        return n_components
    
    def plot_scree(self, save_path: Optional[str] = None):
        """绘制碎石图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 碎石图
        x = range(1, len(self.eigenvalues_) + 1)
        ax1.plot(x, self.eigenvalues_, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=1, color='r', linestyle='--', label='特征值=1')
        ax1.set_xlabel('主成分序号', fontsize=12)
        ax1.set_ylabel('特征值', fontsize=12)
        ax1.set_title('碎石图', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累计方差
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        ax2.bar(x, self.explained_variance_ratio_, alpha=0.7, label='单个贡献率')
        ax2.plot(x, cumulative_variance, 'ro-', linewidth=2, label='累计贡献率')
        ax2.axhline(y=0.85, color='g', linestyle='--', label='85%阈值')
        ax2.set_xlabel('主成分序号', fontsize=12)
        ax2.set_ylabel('方差贡献率', fontsize=12)
        ax2.set_title('方差贡献率', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_biplot(self, X: np.ndarray, labels: List[str] = None,
                   save_path: Optional[str] = None):
        """绘制双标图"""
        # 转换数据
        X_scaled = self.scaler.transform(X) if not self.kernel else X
        scores = self.pca.transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制样本点
        scatter = ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=100, edgecolors='k')
        
        # 添加标签
        if labels is not None:
            for i, label in enumerate(labels):
                ax.annotate(label, (scores[i, 0], scores[i, 1]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=10, fontweight='bold')
        
        # 绘制特征向量
        if self.feature_names is not None:
            scale = min(1.0, 0.5 / np.max(np.abs(self.components_[:2, :])))
            
            for i, feature in enumerate(self.feature_names):
                ax.arrow(0, 0, 
                        self.components_[0, i] * scale,
                        self.components_[1, i] * scale,
                        head_width=0.01, head_length=0.01, 
                        fc='red', ec='red', alpha=0.7)
                
                ax.text(self.components_[0, i] * scale * 1.15,
                       self.components_[1, i] * scale * 1.15,
                       feature, color='red', fontsize=10,
                       ha='center', va='center')
        
        # 设置坐标轴
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA双标图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loadings_heatmap(self, save_path: Optional[str] = None):
        """绘制载荷热力图"""
        if self.loadings_df is None:
            print("没有加载矩阵")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 只显示高载荷（绝对值>0.5）
        loadings_display = self.loadings_df.copy()
        mask = np.abs(loadings_display) < 0.3
        
        sns.heatmap(loadings_display, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, mask=mask, ax=ax,
                   cbar_kws={"shrink": 0.8})
        
        ax.set_title('主成分载荷矩阵（绝对值>0.3）', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_pca_summary(self) -> pd.DataFrame:
        """获取PCA摘要"""
        summary = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(self.eigenvalues_))],
            'Eigenvalue': self.eigenvalues_,
            'Variance_Explained': self.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(self.explained_variance_ratio_)
        })
        
        # 添加主要变量
        if self.loadings_df is not None:
            top_vars = []
            for i in range(len(self.eigenvalues_)):
                pc_loadings = self.loadings_df.iloc[i]
                top_indices = np.abs(pc_loadings).argsort()[-3:][::-1]
                top_vars.append(', '.join([self.feature_names[idx] for idx in top_indices]))
            
            summary['Top_Variables'] = top_vars
        
        return summary
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        X_scaled = self.scaler.transform(X) if not self.kernel else X
        return self.pca.transform(X_scaled)
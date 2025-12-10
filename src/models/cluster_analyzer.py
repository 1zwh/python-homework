import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN,
                           SpectralClustering, OPTICS)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                           davies_bouldin_score, adjusted_rand_score,
                           normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class ClusterAnalyzer:
    """聚类分析器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_centers = None
        self.scores = {}
        
    def determine_optimal_clusters(self, X: np.ndarray, 
                                  method: str = 'kmeans',
                                  max_clusters: int = 10) -> Dict[str, float]:
        """确定最优聚类数量"""
        X_scaled = self.scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]
        
        if method == 'kmeans':
            return self._determine_kmeans_clusters(X_scaled, max_clusters)
        elif method == 'hierarchical':
            return self._determine_hierarchical_clusters(X_scaled, max_clusters)
        elif method == 'gmm':
            return self._determine_gmm_clusters(X_scaled, max_clusters)
        else:
            raise ValueError(f"未知方法: {method}")
    
    def _determine_kmeans_clusters(self, X: np.ndarray, max_clusters: int) -> Dict[str, float]:
        """K-means最优聚类数"""
        scores = {
            'n_clusters': [],
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for n in range(2, min(max_clusters, X.shape[0]) + 1):
            kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) < 2:
                continue
            
            scores['n_clusters'].append(n)
            scores['inertia'].append(kmeans.inertia_)
            scores['silhouette'].append(silhouette_score(X, labels))
            scores['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            scores['davies_bouldin'].append(davies_bouldin_score(X, labels))
        
        return scores
    
    def _determine_hierarchical_clusters(self, X: np.ndarray, max_clusters: int) -> Dict[str, float]:
        """层次聚类最优聚类数"""
        scores = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        linkage_matrix = linkage(X, method='ward')
        
        for n in range(2, min(max_clusters, X.shape[0]) + 1):
            agg = AgglomerativeClustering(n_clusters=n, linkage='ward')
            labels = agg.fit_predict(X)
            
            if len(np.unique(labels)) < 2:
                continue
            
            scores['n_clusters'].append(n)
            scores['silhouette'].append(silhouette_score(X, labels))
            scores['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            scores['davies_bouldin'].append(davies_bouldin_score(X, labels))
        
        return scores
    
    def _determine_gmm_clusters(self, X: np.ndarray, max_clusters: int) -> Dict[str, float]:
        """GMM最优聚类数"""
        scores = {
            'n_clusters': [],
            'bic': [],
            'aic': [],
            'silhouette': []
        }
        
        for n in range(1, min(max_clusters, X.shape[0]) + 1):
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(X)
            labels = gmm.predict(X)
            
            scores['n_clusters'].append(n)
            scores['bic'].append(gmm.bic(X))
            scores['aic'].append(gmm.aic(X))
            
            if len(np.unique(labels)) > 1:
                scores['silhouette'].append(silhouette_score(X, labels))
            else:
                scores['silhouette'].append(np.nan)
        
        return scores
    
    def plot_cluster_evaluation(self, scores: Dict[str, List], 
                              method: str = 'kmeans',
                              save_path: Optional[str] = None):
        """绘制聚类评估图"""
        if method == 'kmeans':
            self._plot_kmeans_evaluation(scores, save_path)
        elif method == 'gmm':
            self._plot_gmm_evaluation(scores, save_path)
        elif method == 'hierarchical':
            self._plot_hierarchical_evaluation(scores, save_path)
    
    def _plot_kmeans_evaluation(self, scores: Dict[str, List], save_path: Optional[str] = None):
        """绘制K-means评估图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 肘部图
        ax1.plot(scores['n_clusters'], scores['inertia'], 'bo-')
        ax1.set_xlabel('聚类数量')
        ax1.set_ylabel('惯量')
        ax1.set_title('肘部图')
        ax1.grid(True, alpha=0.3)
        
        # 轮廓系数
        ax2.plot(scores['n_clusters'], scores['silhouette'], 'ro-')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓系数')
        ax2.grid(True, alpha=0.3)
        
        # Calinski-Harabasz指数
        ax3.plot(scores['n_clusters'], scores['calinski_harabasz'], 'go-')
        ax3.set_xlabel('聚类数量')
        ax3.set_ylabel('Calinski-Harabasz指数')
        ax3.set_title('Calinski-Harabasz指数')
        ax3.grid(True, alpha=0.3)
        
        # Davies-Bouldin指数
        ax4.plot(scores['n_clusters'], scores['davies_bouldin'], 'mo-')
        ax4.set_xlabel('聚类数量')
        ax4.set_ylabel('Davies-Bouldin指数')
        ax4.set_title('Davies-Bouldin指数（越小越好）')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_gmm_evaluation(self, scores: Dict[str, List], save_path: Optional[str] = None):
        """绘制GMM评估图"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # BIC
        ax1.plot(scores['n_clusters'], scores['bic'], 'bo-')
        ax1.set_xlabel('聚类数量')
        ax1.set_ylabel('BIC')
        ax1.set_title('贝叶斯信息准则（BIC）')
        ax1.grid(True, alpha=0.3)
        
        # AIC
        ax2.plot(scores['n_clusters'], scores['aic'], 'ro-')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('AIC')
        ax2.set_title('赤池信息准则（AIC）')
        ax2.grid(True, alpha=0.3)
        
        # 轮廓系数
        ax3.plot(scores['n_clusters'], scores['silhouette'], 'go-')
        ax3.set_xlabel('聚类数量')
        ax3.set_ylabel('轮廓系数')
        ax3.set_title('轮廓系数')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_hierarchical_evaluation(self, scores: Dict[str, List], save_path: Optional[str] = None):
        """绘制层次聚类评估图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 轮廓系数
        ax1.plot(scores['n_clusters'], scores['silhouette'], 'bo-')
        ax1.set_xlabel('聚类数量')
        ax1.set_ylabel('轮廓系数')
        ax1.set_title('轮廓系数')
        ax1.grid(True, alpha=0.3)
        
        # Calinski-Harabasz指数
        ax2.plot(scores['n_clusters'], scores['calinski_harabasz'], 'ro-')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('Calinski-Harabasz指数')
        ax2.set_title('Calinski-Harabasz指数')
        ax2.grid(True, alpha=0.3)
        
        # Davies-Bouldin指数
        ax3.plot(scores['n_clusters'], scores['davies_bouldin'], 'go-')
        ax3.set_xlabel('聚类数量')
        ax3.set_ylabel('Davies-Bouldin指数')
        ax3.set_title('Davies-Bouldin指数')
        ax3.grid(True, alpha=0.3)
        
        # 树状图
        ax4.axis('off')
        ax4.text(0.5, 0.5, '树状图单独绘制', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax4.transAxes,
                fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def fit_clustering(self, X: np.ndarray, 
                      method: str = 'kmeans',
                      n_clusters: Optional[int] = None,
                      **kwargs) -> np.ndarray:
        """拟合聚类模型"""
        X_scaled = self.scaler.fit_transform(X)
        
        if n_clusters is None:
            # 自动确定聚类数量
            if method == 'kmeans':
                scores = self._determine_kmeans_clusters(X_scaled, min(10, X.shape[0]))
                # 使用轮廓系数最大化
                silhouette_scores = scores['silhouette']
                n_clusters = scores['n_clusters'][np.argmax(silhouette_scores)]
            elif method == 'hierarchical':
                scores = self._determine_hierarchical_clusters(X_scaled, min(10, X.shape[0]))
                silhouette_scores = scores['silhouette']
                n_clusters = scores['n_clusters'][np.argmax(silhouette_scores)]
            elif method == 'gmm':
                scores = self._determine_gmm_clusters(X_scaled, min(10, X.shape[0]))
                # 使用BIC最小化
                bic_scores = scores['bic']
                n_clusters = scores['n_clusters'][np.argmin(bic_scores)]
            elif method == 'dbscan':
                n_clusters = None  # DBSCAN自动确定
            else:
                n_clusters = 3
        
        # 应用聚类算法
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, 
                             random_state=self.random_state,
                             n_init=10,
                             **kwargs)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters,
                                              linkage='ward',
                                              **kwargs)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == 'spectral':
            clusterer = SpectralClustering(n_clusters=n_clusters,
                                         random_state=self.random_state,
                                         **kwargs)
        elif method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters,
                                      random_state=self.random_state,
                                      **kwargs)
        elif method == 'optics':
            clusterer = OPTICS(**kwargs)
        else:
            raise ValueError(f"未知聚类方法: {method}")
        
        # 拟合并预测
        if method == 'gmm':
            self.cluster_labels = clusterer.fit_predict(X_scaled)
            self.cluster_centers = clusterer.means_
        elif method == 'kmeans':
            self.cluster_labels = clusterer.fit_predict(X_scaled)
            self.cluster_centers = clusterer.cluster_centers_
        else:
            self.cluster_labels = clusterer.fit_predict(X_scaled)
            self.cluster_centers = None
        
        # 计算评估指标
        if len(np.unique(self.cluster_labels)) > 1:
            self.scores = {
                'silhouette': silhouette_score(X_scaled, self.cluster_labels),
                'calinski_harabasz': calinski_harabasz_score(X_scaled, self.cluster_labels),
                'davies_bouldin': davies_bouldin_score(X_scaled, self.cluster_labels)
            }
        
        return self.cluster_labels
    
    def plot_cluster_results(self, X: np.ndarray, 
                           labels: np.ndarray,
                           feature_names: List[str],
                           save_path: Optional[str] = None):
        """绘制聚类结果"""
        X_scaled = self.scaler.transform(X)
        
        # 使用PCA降维可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 聚类散点图
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # DBSCAN的噪声点
                color = 'gray'
            mask = labels == label
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[color], label=f'Cluster {label}',
                       alpha=0.7, s=100, edgecolors='k')
        
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('聚类可视化（PCA降维）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 聚类特征雷达图（每个聚类的均值）
        if self.cluster_centers is not None and len(feature_names) > 0:
            # 选择前6个特征
            n_features = min(6, len(feature_names))
            selected_features = feature_names[:n_features]
            
            # 计算每个聚类的特征均值
            cluster_means = []
            for label in unique_labels:
                if label != -1:  # 跳过噪声点
                    mask = labels == label
                    if mask.sum() > 0:
                        cluster_means.append(X_scaled[mask][:, :n_features].mean(axis=0))
            
            if cluster_means:
                cluster_means = np.array(cluster_means)
                
                # 创建雷达图
                angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
                angles += angles[:1]  # 闭合图形
                
                ax2 = plt.subplot(2, 2, 2, projection='polar')
                for i, means in enumerate(cluster_means):
                    values = means.tolist()
                    values += values[:1]
                    ax2.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
                    ax2.fill(angles, values, alpha=0.1)
                
                ax2.set_xticks(angles[:-1])
                ax2.set_xticklabels(selected_features)
                ax2.set_title('聚类特征雷达图')
                ax2.legend(bbox_to_anchor=(1.1, 1.05))
        
        # 3. 聚类大小条形图
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        ax3.bar(range(len(cluster_sizes)), cluster_sizes.values)
        ax3.set_xlabel('聚类编号')
        ax3.set_ylabel('样本数量')
        ax3.set_title('聚类规模分布')
        ax3.set_xticks(range(len(cluster_sizes)))
        ax3.set_xticklabels(cluster_sizes.index)
        ax3.grid(True, alpha=0.3)
        
        # 4. 聚类评估指标
        if self.scores:
            metrics = list(self.scores.keys())
            values = list(self.scores.values())
            
            colors = ['skyblue', 'lightgreen', 'lightcoral']
            bars = ax4.bar(metrics, values, color=colors)
            ax4.set_xlabel('评估指标')
            ax4.set_ylabel('分数')
            ax4.set_title('聚类质量评估')
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_cluster_characteristics(self, X: pd.DataFrame, 
                                      labels: np.ndarray) -> pd.DataFrame:
        """分析聚类特征"""
        cluster_stats = []
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # 跳过噪声点
            
            mask = labels == label
            cluster_data = X.iloc[mask]
            
            if len(cluster_data) == 0:
                continue
            
            # 计算统计量
            stats = {
                'cluster': label,
                'n_samples': len(cluster_data),
                'pct_samples': len(cluster_data) / len(X) * 100
            }
            
            # 数值型特征的均值和标准差
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # 只分析前5个特征
                if col in cluster_data.columns:
                    stats[f'{col}_mean'] = cluster_data[col].mean()
                    stats[f'{col}_std'] = cluster_data[col].std()
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
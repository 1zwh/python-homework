import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    import libpysal
    from libpysal import weights
    from esda.moran import Moran, Moran_Local
    from splot.esda import plot_moran, lisa_cluster
    import spreg
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    print("警告: 空间分析库未安装。请运行: pip install geopandas pysal esda splot")

class SpatialModel:
    """空间计量模型"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.weights_matrix = None
        self.moran_results = {}
        self.models = {}
        
    def create_weights_matrix(self, coordinates: np.ndarray, 
                            method: str = 'knn', 
                            k: int = 4,
                            threshold: Optional[float] = None) -> 'weights.W':
        """创建空间权重矩阵"""
        if not HAS_SPATIAL:
            print("空间分析库未安装")
            return None
        
        try:
            if method == 'knn':
                # K近邻权重矩阵
                self.weights_matrix = weights.KNN.from_array(
                    coordinates, k=k, silence_warnings=True
                )
            elif method == 'distance':
                # 距离权重矩阵
                if threshold is None:
                    # 自动确定阈值
                    from scipy.spatial.distance import pdist
                    distances = pdist(coordinates)
                    threshold = np.percentile(distances, 25)
                
                self.weights_matrix = weights.DistanceBand.from_array(
                    coordinates, threshold=threshold, binary=False, alpha=-1.0
                )
            elif method == 'queen':
                # 假设我们有面数据（这里简化为基于距离的近似）
                self.weights_matrix = weights.KNN.from_array(
                    coordinates, k=k, silence_warnings=True
                )
            
            # 标准化权重矩阵
            self.weights_matrix.transform = 'r'
            
            return self.weights_matrix
            
        except Exception as e:
            print(f"创建权重矩阵时出错: {e}")
            return None
    
    def calculate_moran_i(self, y: np.ndarray, 
                         w: Optional['weights.W'] = None,
                         permutations: int = 999) -> Dict:
        """计算莫兰指数"""
        if not HAS_SPATIAL:
            return {'error': '空间分析库未安装'}
        
        if w is None:
            w = self.weights_matrix
        
        if w is None:
            return {'error': '权重矩阵未定义'}
        
        try:
            moran = Moran(y, w, permutations=permutations)
            
            results = {
                'I': moran.I,
                'EI': moran.EI,
                'VI_norm': moran.VI_norm,
                'z_norm': moran.z_norm,
                'p_norm': moran.p_norm,
                'z_rand': moran.z_rand,
                'p_rand': moran.p_rand,
                'z_sim': moran.z_sim,
                'p_sim': moran.p_sim,
                'significant': moran.p_sim < 0.05
            }
            
            self.moran_results['global'] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_local_moran(self, y: np.ndarray,
                            w: Optional['weights.W'] = None,
                            permutations: int = 999) -> Dict:
        """计算局部莫兰指数"""
        if not HAS_SPATIAL:
            return {'error': '空间分析库未安装'}
        
        if w is None:
            w = self.weights_matrix
        
        if w is None:
            return {'error': '权重矩阵未定义'}
        
        try:
            lisa = Moran_Local(y, w, permutations=permutations)
            
            results = {
                'Is': lisa.Is,
                'p_sim': lisa.p_sim,
                'q': lisa.q,
                'clusters': self._interpret_lisa_clusters(lisa)
            }
            
            self.moran_results['local'] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _interpret_lisa_clusters(self, lisa) -> List[str]:
        """解释LISA聚类类型"""
        clusters = []
        for i in range(len(lisa.Is)):
            if lisa.p_sim[i] > 0.05:
                clusters.append('不显著')
            elif lisa.q[i] == 1:
                clusters.append('高-高')
            elif lisa.q[i] == 2:
                clusters.append('低-高')
            elif lisa.q[i] == 3:
                clusters.append('低-低')
            elif lisa.q[i] == 4:
                clusters.append('高-低')
            else:
                clusters.append('未知')
        return clusters
    
    def fit_spatial_lag_model(self, y: np.ndarray, X: np.ndarray,
                            w: Optional['weights.W'] = None,
                            name: str = 'SLM') -> Dict:
        """拟合空间滞后模型(SLM)"""
        if not HAS_SPATIAL:
            return {'error': '空间分析库未安装'}
        
        if w is None:
            w = self.weights_matrix
        
        if w is None:
            return {'error': '权重矩阵未定义'}
        
        try:
            # 简化实现 - 使用OLS估计
            from sklearn.linear_model import LinearRegression
            
            # 创建空间滞后变量
            wy = w.sparse.dot(y.reshape(-1, 1)).ravel()
            X_with_lag = np.column_stack([X, wy])
            
            # 拟合模型
            model = LinearRegression()
            model.fit(X_with_lag, y)
            
            # 计算指标
            y_pred = model.predict(X_with_lag)
            r2 = model.score(X_with_lag, y)
            mse = np.mean((y - y_pred) ** 2)
            
            results = {
                'model_type': 'SLM',
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r2': r2,
                'mse': mse,
                'nobs': len(y),
                'nvar': X.shape[1] + 1
            }
            
            self.models[name] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def fit_spatial_error_model(self, y: np.ndarray, X: np.ndarray,
                              w: Optional['weights.W'] = None,
                              name: str = 'SEM') -> Dict:
        """拟合空间误差模型(SEM)"""
        if not HAS_SPATIAL:
            return {'error': '空间分析库未安装'}
        
        # 简化实现 - 返回基本信息
        results = {
            'model_type': 'SEM',
            'note': '完整SEM实现需要更多依赖',
            'nobs': len(y),
            'nvar': X.shape[1]
        }
        
        self.models[name] = results
        return results
    
    def fit_spatial_durbin_model(self, y: np.ndarray, X: np.ndarray,
                               w: Optional['weights.W'] = None,
                               name: str = 'SDM') -> Dict:
        """拟合空间杜宾模型(SDM)"""
        if not HAS_SPATIAL:
            return {'error': '空间分析库未安装'}
        
        # 简化实现 - 返回基本信息
        results = {
            'model_type': 'SDM',
            'note': '完整SDM实现需要更多依赖',
            'nobs': len(y),
            'nvar': X.shape[1]
        }
        
        self.models[name] = results
        return results
    
    def perform_spatial_regression(self, y: np.ndarray, X: np.ndarray,
                                 w: Optional['weights.W'] = None,
                                 model_types: List[str] = None) -> Dict:
        """执行空间回归分析"""
        if model_types is None:
            model_types = ['SLM', 'SEM', 'SDM']
        
        results = {}
        
        for model_type in model_types:
            if model_type == 'SLM':
                results['SLM'] = self.fit_spatial_lag_model(y, X, w, 'SLM')
            elif model_type == 'SEM':
                results['SEM'] = self.fit_spatial_error_model(y, X, w, 'SEM')
            elif model_type == 'SDM':
                results['SDM'] = self.fit_spatial_durbin_model(y, X, w, 'SDM')
        
        return results
    
    def get_model_summary(self) -> pd.DataFrame:
        """获取模型摘要"""
        summaries = []
        
        for name, model in self.models.items():
            if 'error' not in model:
                summary = {
                    '模型': name,
                    '类型': model.get('model_type', '未知'),
                    '样本数': model.get('nobs', 0),
                    '变量数': model.get('nvar', 0),
                    'R平方': model.get('r2', np.nan),
                    'MSE': model.get('mse', np.nan)
                }
                summaries.append(summary)
        
        return pd.DataFrame(summaries)
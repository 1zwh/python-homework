import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import PolynomialFeatures
import warnings

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.poly_transformer = None
        self.interaction_features = []
        
    def create_lag_features(self, df: pd.DataFrame, 
                          value_cols: List[str], 
                          group_col: str = 'city', 
                          time_col: str = '年份',
                          lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """创建滞后特征"""
        df_lagged = df.copy()
        df_lagged = df_lagged.sort_values([group_col, time_col])
        
        for col in value_cols:
            for lag in lags:
                df_lagged[f'{col}_lag{lag}'] = df_lagged.groupby(group_col)[col].shift(lag)
        
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame,
                              value_cols: List[str],
                              group_col: str = 'city',
                              time_col: str = '年份',
                              windows: List[int] = [3, 5]) -> pd.DataFrame:
        """创建滚动统计特征"""
        df_rolling = df.copy()
        df_rolling = df_rolling.sort_values([group_col, time_col])
        
        for col in value_cols:
            for window in windows:
                # 滚动均值
                df_rolling[f'{col}_rolling_mean_{window}'] = (
                    df_rolling.groupby(group_col)[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # 滚动标准差
                df_rolling[f'{col}_rolling_std_{window}'] = (
                    df_rolling.groupby(group_col)[col]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                
                # 滚动最大值
                df_rolling[f'{col}_rolling_max_{window}'] = (
                    df_rolling.groupby(group_col)[col]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(level=0, drop=True)
                )
        
        return df_rolling
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                 feature_cols: List[str],
                                 degree: int = 2,
                                 interaction_only: bool = False) -> pd.DataFrame:
        """创建多项式特征"""
        df_poly = df.copy()
        
        # 选择数值型特征
        X = df_poly[feature_cols].values
        
        # 创建多项式特征
        self.poly_transformer = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        X_poly = self.poly_transformer.fit_transform(X)
        
        # 获取特征名称
        feature_names = self.poly_transformer.get_feature_names_out(feature_cols)
        
        # 添加到DataFrame
        poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df_poly.index)
        
        # 合并回原数据
        df_poly = pd.concat([df_poly, poly_df], axis=1)
        
        return df_poly
    
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """创建交互特征"""
        df_interaction = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df_interaction.columns and col2 in df_interaction.columns:
                interaction_name = f'{col1}_x_{col2}'
                df_interaction[interaction_name] = df_interaction[col1] * df_interaction[col2]
                self.interaction_features.append(interaction_name)
        
        return df_interaction
    
    def create_ratio_features(self, df: pd.DataFrame,
                            numerator_cols: List[str],
                            denominator_cols: List[str]) -> pd.DataFrame:
        """创建比率特征"""
        df_ratio = df.copy()
        
        for num in numerator_cols:
            for denom in denominator_cols:
                if num in df_ratio.columns and denom in df_ratio.columns:
                    # 避免除零
                    denominator = df_ratio[denom].replace(0, np.nan)
                    ratio_name = f'{num}_per_{denom}'
                    df_ratio[ratio_name] = df_ratio[num] / denominator
        
        return df_ratio
    
    def create_growth_features(self, df: pd.DataFrame,
                             value_cols: List[str],
                             group_col: str = 'city',
                             time_col: str = '年份') -> pd.DataFrame:
        """创建增长率特征"""
        df_growth = df.copy()
        df_growth = df_growth.sort_values([group_col, time_col])
        
        for col in value_cols:
            # 计算同比变化
            df_growth[f'{col}_growth'] = df_growth.groupby(group_col)[col].pct_change()
            
            # 计算年度变化
            df_growth[f'{col}_yoy'] = df_growth.groupby(group_col)[col].pct_change(periods=1)
        
        return df_growth
    
    def create_statistical_features(self, df: pd.DataFrame,
                                  value_cols: List[str],
                                  group_col: str = 'city') -> pd.DataFrame:
        """创建统计特征"""
        df_stats = df.copy()
        
        for col in value_cols:
            if col in df_stats.columns:
                # 分组统计
                group_stats = df_stats.groupby(group_col)[col].agg([
                    'mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'
                ]).add_prefix(f'{col}_')
                
                # 合并回原数据
                df_stats = df_stats.merge(group_stats, on=group_col, how='left')
        
        return df_stats
    
    def create_distance_features(self, df: pd.DataFrame,
                               value_cols: List[str],
                               city_col: str = 'city',
                               method: str = 'euclidean') -> pd.DataFrame:
        """创建距离特征（城市间相似度）"""
        df_distance = df.copy()
        
        # 按城市分组，取最新年份
        latest_year = df_distance['年份'].max()
        df_latest = df_distance[df_distance['年份'] == latest_year].copy()
        
        if len(df_latest) < 2:
            return df_distance
        
        # 创建城市特征矩阵
        cities = df_latest[city_col].unique()
        feature_matrix = []
        
        for city in cities:
            city_data = df_latest[df_latest[city_col] == city]
            if not city_data.empty:
                city_features = city_data[value_cols].values[0]
                feature_matrix.append(city_features)
        
        feature_matrix = np.array(feature_matrix)
        
        # 计算距离矩阵
        distance_matrix = squareform(pdist(feature_matrix, metric=method))
        
        # 为每个城市找到最近邻
        np.fill_diagonal(distance_matrix, np.inf)  # 忽略自身
        nearest_indices = np.argmin(distance_matrix, axis=1)
        
        # 计算与最近邻的距离和相似度
        similarity_features = []
        for i, city in enumerate(cities):
            nearest_idx = nearest_indices[i]
            distance = distance_matrix[i, nearest_idx]
            nearest_city = cities[nearest_idx]
            
            similarity_features.append({
                city_col: city,
                f'distance_to_nearest_{method}': distance,
                'nearest_city': nearest_city
            })
        
        similarity_df = pd.DataFrame(similarity_features)
        df_distance = df_distance.merge(similarity_df, on=city_col, how='left')
        
        return df_distance
    
    def create_network_position_features(self, od_data: pd.DataFrame,
                                       city_col: str = 'city') -> pd.DataFrame:
        """基于OD数据创建网络位置特征"""
        # 计算度中心性
        out_degree = od_data.groupby('source_city').size()
        in_degree = od_data.groupby('target_city').size()
        
        # 计算强度中心性
        out_strength = od_data.groupby('source_city')['value'].sum()
        in_strength = od_data.groupby('target_city')['value'].sum()
        
        # 创建特征DataFrame
        network_features = pd.DataFrame({
            'out_degree': out_degree,
            'in_degree': in_degree,
            'total_degree': out_degree + in_degree,
            'out_strength': out_strength,
            'in_strength': in_strength,
            'total_strength': out_strength + in_strength,
            'net_flow': out_strength - in_strength
        }).reset_index()
        
        network_features = network_features.rename(columns={'index': city_col})
        
        # 计算中介中心性（简化版）
        city_codes = od_data[['source_city', 'target_city']].values.flatten()
        unique_cities = np.unique(city_codes)
        
        betweenness = {}
        for city in unique_cities:
            # 简化计算：该城市在OD对中出现的频率
            betweenness[city] = len(od_data[
                (od_data['source_city'] == city) | 
                (od_data['target_city'] == city)
            ]) / len(od_data)
        
        network_features['betweenness_centrality'] = network_features[city_col].map(betweenness)
        
        return network_features
    
    def full_feature_engineering_pipeline(self, df: pd.DataFrame,
                                        od_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """完整的特征工程流程"""
        print("开始特征工程流程...")
        df_engineered = df.copy()
        
        # 识别数值型特征
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [col for col in numeric_cols if col not in ['年份', 'city_code']]
        
        if len(value_cols) == 0:
            print("没有找到数值型特征")
            return df_engineered
        
        print(f"找到{len(value_cols)}个数值型特征")
        
        # 1. 创建滞后特征
        print("\n1. 创建滞后特征...")
        df_engineered = self.create_lag_features(df_engineered, value_cols[:5])
        
        # 2. 创建滚动特征
        print("2. 创建滚动特征...")
        df_engineered = self.create_rolling_features(df_engineered, value_cols[:5])
        
        # 3. 创建增长率特征
        print("3. 创建增长率特征...")
        df_engineered = self.create_growth_features(df_engineered, value_cols[:5])
        
        # 4. 创建统计特征
        print("4. 创建统计特征...")
        df_engineered = self.create_statistical_features(df_engineered, value_cols[:5])
        
        # 5. 创建交互特征（选择关键特征对）
        print("5. 创建交互特征...")
        if len(value_cols) >= 4:
            feature_pairs = [
                (value_cols[0], value_cols[1]),
                (value_cols[2], value_cols[3])
            ]
            df_engineered = self.create_interaction_features(df_engineered, feature_pairs)
        
        # 6. 创建比率特征
        print("6. 创建比率特征...")
        if len(value_cols) >= 4:
            df_engineered = self.create_ratio_features(
                df_engineered, 
                value_cols[:2], 
                value_cols[2:4]
            )
        
        # 7. 创建距离特征
        print("7. 创建距离特征...")
        df_engineered = self.create_distance_features(df_engineered, value_cols[:5])
        
        # 8. 创建网络特征（如果有OD数据）
        if od_data is not None and 'city' in df_engineered.columns:
            print("8. 创建网络特征...")
            network_features = self.create_network_position_features(od_data)
            df_engineered = df_engineered.merge(network_features, on='city', how='left')
        
        print(f"\n特征工程完成!")
        print(f"原始特征数: {len(df.columns)}")
        print(f"新增特征数: {len(df_engineered.columns) - len(df.columns)}")
        print(f"总特征数: {len(df_engineered.columns)}")
        
        return df_engineered
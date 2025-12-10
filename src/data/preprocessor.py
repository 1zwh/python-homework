import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
import warnings

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, method: str = 'mean', scale_method: str = 'standard'):
        """
        初始化预处理器
        
        Parameters
        ----------
        method : str
            缺失值处理方法 ('mean', 'median', 'knn', 'mice')
        scale_method : str
            标准化方法 ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scale_method = scale_method
        self.imputer = None
        self.scaler = None
        self.imputer_fitted = False
        self.scaler_fitted = False
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = None) -> pd.DataFrame:
        """处理缺失值"""
        if strategy is None:
            strategy = self.method
            
        df_clean = df.copy()
        
        # 检查缺失值
        missing_sum = df_clean.isnull().sum()
        missing_cols = missing_sum[missing_sum > 0]
        
        if len(missing_cols) == 0:
            print("没有缺失值")
            return df_clean
        
        print(f"缺失值列数: {len(missing_cols)}")
        print(f"总缺失值数: {missing_sum.sum()}")
        
        # 根据策略选择方法
        if strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        elif strategy == 'mice':
            self.imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            raise ValueError(f"未知的缺失值处理方法: {strategy}")
        
        # 应用插补
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_numeric = df_clean[numeric_cols]
        
        df_imputed = self.imputer.fit_transform(df_numeric)
        df_clean[numeric_cols] = df_imputed
        
        self.imputer_fitted = True
        print(f"使用{strategy}方法完成缺失值处理")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """检测异常值"""
        df_clean = df.copy()
        outliers_report = {}
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df_clean[col].dropna()
            
            if method == 'iqr':
                # IQR方法
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                
            elif method == 'zscore':
                # Z-score方法
                z_scores = np.abs(stats.zscore(data))
                outliers = (z_scores > threshold).sum()
                
            elif method == 'modified_zscore':
                # 修正Z-score方法
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else 0
                outliers = (np.abs(modified_z_scores) > threshold).sum()
            
            outliers_report[col] = {
                'outliers_count': outliers,
                'outliers_pct': outliers / len(data) * 100 if len(data) > 0 else 0
            }
        
        # 创建异常值报告
        outliers_df = pd.DataFrame(outliers_report).T
        outliers_df = outliers_df.sort_values('outliers_pct', ascending=False)
        
        print(f"异常值检测完成 (方法: {method})")
        print(f"异常值最多的5列:")
        print(outliers_df.head())
        
        return outliers_df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap', threshold: float = 3) -> pd.DataFrame:
        """处理异常值"""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df_clean[col].dropna()
            
            if len(data) < 10:  # 数据太少不处理
                continue
            
            # 计算边界
            if method == 'cap':
                # 盖帽法
                percentile_low = np.percentile(data, 1)
                percentile_high = np.percentile(data, 99)
                
                df_clean[col] = df_clean[col].clip(percentile_low, percentile_high)
                
            elif method == 'remove':
                # 删除异常值（设置为NaN）
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
                
            elif method == 'transform':
                # 使用对数变换
                if (df_clean[col] > 0).all():
                    df_clean[col] = np.log1p(df_clean[col])
                else:
                    # 如果有非正值，先平移
                    min_val = df_clean[col].min()
                    if min_val <= 0:
                        df_clean[col] = np.log1p(df_clean[col] - min_val + 1)
        
        print(f"异常值处理完成 (方法: {method})")
        return df_clean
    
    def scale_data(self, df: pd.DataFrame, method: str = None) -> pd.DataFrame:
        """数据标准化"""
        if method is None:
            method = self.scale_method
            
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"未知的标准化方法: {method}")
        
        df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        self.scaler_fitted = True
        
        print(f"数据标准化完成 (方法: {method})")
        return df_scaled
    
    def check_normality(self, df: pd.DataFrame) -> pd.DataFrame:
        """检验正态性"""
        normality_report = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            if len(data) < 8:  # Shapiro-Wilk检验需要至少8个样本
                continue
            
            # Shapiro-Wilk检验
            stat_sw, p_sw = stats.shapiro(data)
            
            # Kolmogorov-Smirnov检验
            stat_ks, p_ks = stats.kstest(data, 'norm', 
                                       args=(data.mean(), data.std()))
            
            # 偏度和峰度
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            normality_report.append({
                'variable': col,
                'shapiro_stat': stat_sw,
                'shapiro_p': p_sw,
                'ks_stat': stat_ks,
                'ks_p': p_ks,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': p_sw > 0.05 and abs(skewness) < 2 and abs(kurtosis) < 7
            })
        
        report_df = pd.DataFrame(normality_report)
        return report_df
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用数据变换"""
        df_transformed = df.copy()
        
        normality_report = self.check_normality(df_transformed)
        non_normal_cols = normality_report[~normality_report['is_normal']]['variable'].tolist()
        
        for col in non_normal_cols:
            if col in df_transformed.columns:
                data = df_transformed[col].dropna()
                
                # 尝试不同变换
                if (data > 0).all():
                    # 对数变换
                    df_transformed[f'{col}_log'] = np.log(data)
                
                # 平方根变换
                if (data >= 0).all():
                    df_transformed[f'{col}_sqrt'] = np.sqrt(data)
                
                # Box-Cox变换（需要正值）
                if (data > 0).all() and len(data) > 10:
                    try:
                        transformed, _ = stats.boxcox(data)
                        df_transformed[f'{col}_boxcox'] = transformed
                    except:
                        pass
        
        print(f"数据变换完成，新增{len(df_transformed.columns) - len(df.columns)}个变换特征")
        return df_transformed
    
    def full_preprocessing_pipeline(self, df: pd.DataFrame, 
                                   handle_missing: bool = True,
                                   detect_outliers: bool = True,
                                   handle_outliers: bool = True,
                                   scale: bool = True,
                                   transform: bool = True) -> pd.DataFrame:
        """完整的数据预处理流程"""
        print("开始数据预处理流程...")
        df_processed = df.copy()
        
        # 1. 处理缺失值
        if handle_missing:
            print("\n1. 处理缺失值...")
            df_processed = self.handle_missing_values(df_processed)
        
        # 2. 检测异常值
        if detect_outliers:
            print("\n2. 检测异常值...")
            outliers_report = self.detect_outliers(df_processed)
        
        # 3. 处理异常值
        if handle_outliers:
            print("\n3. 处理异常值...")
            df_processed = self.handle_outliers(df_processed)
        
        # 4. 数据变换
        if transform:
            print("\n4. 数据变换...")
            df_processed = self.apply_transformations(df_processed)
        
        # 5. 标准化
        if scale:
            print("\n5. 数据标准化...")
            df_processed = self.scale_data(df_processed)
        
        print("\n数据预处理完成!")
        print(f"原始形状: {df.shape}")
        print(f"处理后形状: {df_processed.shape}")
        
        return df_processed
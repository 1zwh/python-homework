import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    """机器学习预测器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2) -> Tuple:
        """准备数据"""
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                              model_name: str = 'linear_regression') -> Dict:
        """训练线性回归"""
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=self.random_state),
                'Lasso': Lasso(random_state=self.random_state),
                'ElasticNet': ElasticNet(random_state=self.random_state)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                self.models[f'{model_name}_{name}'] = model
                results[name] = {
                    'coefficients': model.coef_ if hasattr(model, 'coef_') else None,
                    'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def train_tree_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         model_name: str = 'tree_models') -> Dict:
        """训练树模型"""
        try:
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            
            models = {
                'DecisionTree': DecisionTreeRegressor(random_state=self.random_state),
                'RandomForest': RandomForestRegressor(random_state=self.random_state, n_estimators=100),
                'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state, n_estimators=100)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                self.models[f'{model_name}_{name}'] = model
                results[name] = {
                    'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     model_name: str = 'xgboost') -> Dict:
        """训练XGBoost模型"""
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
            
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            return {
                'feature_importance': model.feature_importances_,
                'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
            }
            
        except ImportError:
            return {'error': 'XGBoost未安装'}
        except Exception as e:
            return {'error': str(e)}
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      model_name: str = 'lightgbm') -> Dict:
        """训练LightGBM模型"""
        try:
            import lightgbm as lgb
            
            model = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
            
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            return {
                'feature_importance': model.feature_importances_,
                'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else None
            }
            
        except ImportError:
            return {'error': 'LightGBM未安装'}
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型"""
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if np.all(y_test != 0) else None
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray,
                        cv: int = 5) -> Dict:
        """交叉验证"""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return {
                'cv_scores': scores.tolist(),
                'cv_mean': np.mean(scores),
                'cv_std': np.std(scores)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def hyperparameter_tuning(self, model, param_grid: Dict,
                            X_train: np.ndarray, y_train: np.ndarray,
                            cv: int = 5) -> Dict:
        """超参数调优"""
        try:
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='r2',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2) -> Dict:
        """训练所有模型"""
        # 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size)
        
        all_results = {}
        
        # 训练不同类型模型
        print("训练线性回归模型...")
        linear_results = self.train_linear_regression(X_train, y_train)
        all_results['linear'] = linear_results
        
        print("训练树模型...")
        tree_results = self.train_tree_models(X_train, y_train)
        all_results['tree'] = tree_results
        
        print("训练XGBoost...")
        xgb_results = self.train_xgboost(X_train, y_train)
        all_results['xgboost'] = xgb_results
        
        print("训练LightGBM...")
        lgb_results = self.train_lightgbm(X_train, y_train)
        all_results['lightgbm'] = lgb_results
        
        # 评估所有模型
        print("评估模型...")
        evaluation_results = {}
        for name, model in self.models.items():
            eval_metrics = self.evaluate_model(model, X_test, y_test)
            evaluation_results[name] = eval_metrics
            
            # 交叉验证
            cv_results = self.cross_validation(model, X_train, y_train)
            evaluation_results[name]['cv'] = cv_results
        
        all_results['evaluation'] = evaluation_results
        self.results = all_results
        
        return all_results
    
    def get_best_model(self) -> Tuple:
        """获取最佳模型"""
        if not self.results or 'evaluation' not in self.results:
            return None, None
        
        evaluation = self.results['evaluation']
        best_model_name = None
        best_r2 = -np.inf
        
        for model_name, metrics in evaluation.items():
            if 'error' not in metrics and metrics.get('r2', -np.inf) > best_r2:
                best_r2 = metrics['r2']
                best_model_name = model_name
        
        if best_model_name and best_model_name in self.models:
            return self.models[best_model_name], best_model_name
        
        return None, None
    
    def get_model_summary(self) -> pd.DataFrame:
        """获取模型摘要"""
        summaries = []
        
        if 'evaluation' in self.results:
            for model_name, metrics in self.results['evaluation'].items():
                if 'error' not in metrics:
                    summary = {
                        '模型': model_name,
                        'R2': metrics.get('r2', np.nan),
                        'RMSE': metrics.get('rmse', np.nan),
                        'MAE': metrics.get('mae', np.nan),
                        '交叉验证R2均值': metrics.get('cv', {}).get('cv_mean', np.nan),
                        '交叉验证R2标准差': metrics.get('cv', {}).get('cv_std', np.nan)
                    }
                    summaries.append(summary)
        
        return pd.DataFrame(summaries).sort_values('R2', ascending=False)
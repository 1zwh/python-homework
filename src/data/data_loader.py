import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings

class DataLoader:
    """数据加载器"""
    
    def __init__(self, config_path: str = "config/data_config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.city_mapping = self.config['city_mapping']
        self.reverse_city_mapping = {v: k for k, v in self.city_mapping.items()}
        self.raw_path = "data/raw/"
        
    def load_all_od_data(self) -> pd.DataFrame:
        """加载所有OD数据"""
        try:
            # 加载主OD文件
            od_main = pd.read_csv(os.path.join(self.raw_path, "od_matrix.csv"))
            od_main = od_main.rename(columns={
                '数据传输量_TB': 'value',
                'API调用频次_万次': 'api_calls',
                '企业合作数据项目数': 'cooperation_projects'
            })
            
            # 加载各年度OD矩阵并合并
            od_dfs = [od_main]
            
            for year in [2019, 2020, 2021, 2022, 2023]:
                try:
                    file_name = f"od_matrix_{year}.csv"
                    file_path = os.path.join(self.raw_path, file_name)
                    
                    if os.path.exists(file_path):
                        df_year = pd.read_csv(file_path)
                        
                        # 转换宽表为长表
                        melted = df_year.melt(
                            id_vars=['起点城市'],
                            var_name='终点城市',
                            value_name='value'
                        )
                        melted['年份'] = year
                        melted['数据类型'] = '传输量'
                        
                        od_dfs.append(melted)
                except Exception as e:
                    print(f"加载{year}年数据时出错: {e}")
            
            # 合并所有数据
            od_combined = pd.concat(od_dfs, ignore_index=True)
            
            # 统一列名
            if '起点城市' in od_combined.columns:
                od_combined['source_city'] = od_combined['起点城市']
                od_combined['target_city'] = od_combined['终点城市']
            elif 'source_city' not in od_combined.columns:
                # 如果主OD文件有不同列名
                if '起点城市' in od_main.columns:
                    od_combined['source_city'] = od_combined['起点城市']
                    od_combined['target_city'] = od_combined['终点城市']
            
            # 添加城市代码
            od_combined['source_code'] = od_combined['source_city'].map(self.city_mapping)
            od_combined['target_code'] = od_combined['target_city'].map(self.city_mapping)
            
            # 填充缺失的年份（主文件可能没有年份列）
            if '年份' not in od_combined.columns:
                od_combined['年份'] = 2019  # 假设主文件是2019年
            
            # 去除缺失值
            od_combined = od_combined.dropna(subset=['value', 'source_city', 'target_city'])
            
            print(f"加载完成: {len(od_combined)}条OD记录")
            return od_combined
            
        except Exception as e:
            print(f"加载OD数据时出错: {e}")
            return pd.DataFrame()
    
    def load_panel_data(self) -> pd.DataFrame:
        """加载面板数据"""
        try:
            file_path = os.path.join(self.raw_path, "main_data_advanced.csv")
            panel_data = pd.read_csv(file_path)
            
            # 重命名列以去除特殊字符
            panel_data.columns = [col.replace(' ', '_').replace('%', 'pct') for col in panel_data.columns]
            
            # 添加城市代码
            panel_data['city_code'] = panel_data['城市'].map(self.city_mapping)
            
            print(f"加载完成: {len(panel_data)}条面板记录")
            return panel_data
            
        except Exception as e:
            print(f"加载面板数据时出错: {e}")
            return pd.DataFrame()
    
    def create_city_year_matrix(self, od_data: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
        """创建城市-年份矩阵"""
        try:
            # 按城市和年份汇总
            city_year_matrix = od_data.groupby(['source_city', '年份'])[value_col].sum().reset_index()
            
            # 转换为宽表
            matrix_wide = city_year_matrix.pivot(
                index='source_city',
                columns='年份',
                values=value_col
            ).fillna(0)
            
            return matrix_wide
            
        except Exception as e:
            print(f"创建城市-年份矩阵时出错: {e}")
            return pd.DataFrame()
    
    def get_city_characteristics(self, panel_data: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
        """获取指定年份的城市特征"""
        try:
            year_data = panel_data[panel_data['年份'] == year].copy()
            
            # 选择关键指标
            selected_vars = []
            for category in self.config['analysis_vars'].values():
                selected_vars.extend(category)
            
            # 只保留存在的列
            existing_vars = [var for var in selected_vars if var in year_data.columns]
            
            characteristics = year_data[['城市', 'city_code'] + existing_vars].copy()
            
            return characteristics
            
        except Exception as e:
            print(f"获取城市特征时出错: {e}")
            return pd.DataFrame()
    
    def create_combined_dataset(self, year: int = 2023) -> pd.DataFrame:
        """创建合并数据集（OD数据 + 城市特征）"""
        try:
            od_data = self.load_all_od_data()
            panel_data = self.load_panel_data()
            
            # 获取OD网络特征
            network_features = self.calculate_network_features(od_data, year)
            
            # 获取城市特征
            city_features = self.get_city_characteristics(panel_data, year)
            
            # 合并数据
            combined = pd.merge(
                network_features,
                city_features,
                left_on='city',
                right_on='城市',
                how='left'
            ).drop('城市', axis=1)
            
            print(f"合并完成: {len(combined)}个观测值")
            return combined
            
        except Exception as e:
            print(f"创建合并数据集时出错: {e}")
            return pd.DataFrame()
    
    def calculate_network_features(self, od_data: pd.DataFrame, year: int) -> pd.DataFrame:
        """计算网络特征"""
        try:
            year_data = od_data[od_data['年份'] == year]
            
            features = []
            cities = list(self.city_mapping.keys())
            
            for city in cities:
                # 流出特征
                outflow = year_data[year_data['source_city'] == city]
                outflow_strength = outflow['value'].sum() if not outflow.empty else 0
                outflow_degree = len(outflow) if not outflow.empty else 0
                
                # 流入特征
                inflow = year_data[year_data['target_city'] == city]
                inflow_strength = inflow['value'].sum() if not inflow.empty else 0
                inflow_degree = len(inflow) if not inflow.empty else 0
                
                # 网络位置特征
                total_strength = outflow_strength + inflow_strength
                net_flow = outflow_strength - inflow_strength
                flow_balance = abs(net_flow) / total_strength if total_strength > 0 else 0
                
                features.append({
                    '城市': city,  # 改为中文列名
                    'city_code': self.city_mapping[city],
                    'outflow_strength': outflow_strength,
                    'inflow_strength': inflow_strength,
                    'total_strength': total_strength,
                    'net_flow': net_flow,
                    'flow_balance': flow_balance,
                    'out_degree': outflow_degree,
                    'in_degree': inflow_degree,
                    'total_degree': outflow_degree + inflow_degree
                })
            
            return pd.DataFrame(features)
            
        except Exception as e:
            print(f"计算网络特征时出错: {e}")
            return pd.DataFrame()
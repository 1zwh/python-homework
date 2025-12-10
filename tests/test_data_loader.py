import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataLoader

class TestDataLoader:
    """测试数据加载器"""
    
    def setup_method(self):
        """测试设置"""
        self.loader = DataLoader()
    
    def test_city_mapping(self):
        """测试城市映射"""
        assert self.loader.city_mapping['香港'] == 1
        assert self.loader.city_mapping['广州'] == 3
        assert self.loader.city_mapping['深圳'] == 4
        assert len(self.loader.city_mapping) == 11
    
    def test_load_all_od_data(self):
        """测试加载OD数据"""
        od_data = self.loader.load_all_od_data()
        
        # 检查数据不为空
        assert not od_data.empty
        
        # 检查必要的列存在
        required_columns = ['source_city', 'target_city', 'value', '年份']
        for col in required_columns:
            assert col in od_data.columns
        
        # 检查城市数量
        unique_cities = pd.concat([od_data['source_city'], od_data['target_city']]).unique()
        assert len(unique_cities) >= 10  # 应该有11个城市
    
    def test_load_panel_data(self):
        """测试加载面板数据"""
        panel_data = self.loader.load_panel_data()
        
        # 检查数据不为空
        assert not panel_data.empty
        
        # 检查必要的列存在
        required_columns = ['城市', '年份', 'GDP_亿元']
        for col in required_columns:
            assert col in panel_data.columns
        
        # 检查年份范围
        years = panel_data['年份'].unique()
        assert len(years) >= 1
    
    def test_create_city_year_matrix(self):
        """测试创建城市-年份矩阵"""
        od_data = self.loader.load_all_od_data()
        
        if not od_data.empty:
            matrix = self.loader.create_city_year_matrix(od_data)
            
            # 检查矩阵不为空
            assert not matrix.empty
            
            # 检查形状
            assert matrix.shape[0] >= 10  # 行数为城市数
            assert matrix.shape[1] >= 1   # 列数为年份数
    
    def test_get_city_characteristics(self):
        """测试获取城市特征"""
        panel_data = self.loader.load_panel_data()
        
        if not panel_data.empty:
            characteristics = self.loader.get_city_characteristics(panel_data, 2023)
            
            # 检查数据不为空
            assert not characteristics.empty
            
            # 检查必要的列存在
            assert '城市' in characteristics.columns
            assert 'city_code' in characteristics.columns
    
    def test_calculate_network_features(self):
        """测试计算网络特征"""
        od_data = self.loader.load_all_od_data()
        
        if not od_data.empty:
            # 选择一个存在的年份
            available_year = od_data['年份'].iloc[0]
            features = self.loader.calculate_network_features(od_data, available_year)
            
            # 检查数据不为空
            assert not features.empty
            
            # 检查必要的列存在
            required_columns = ['city', 'outflow_strength', 'inflow_strength', 'total_strength']
            for col in required_columns:
                assert col in features.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
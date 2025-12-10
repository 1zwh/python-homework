import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from typing import Optional, List

# 解决中文显示问题（关键：避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class VisualizationUtils:
    """空间可视化工具类"""
    
    def plot_spatial_distribution(self, df: pd.DataFrame,
            value_col: str,
            city_col: str = '城市',  # 中文列名
            year: Optional[int] = None,
            title: str = "空间分布图",
            save_path: Optional[str] = None):
        """
        绘制粤港澳大湾区城市空间分布热力图
        
        参数：
        df: pd.DataFrame - 包含城市、年份、数值的数据集
        value_col: str - 要可视化的数值列名
        city_col: str - 城市列名（默认：'城市'）
        year: Optional[int] - 筛选年份（None则不筛选）
        title: str - 图表标题
        save_path: Optional[str] - 保存路径（None则不保存）
        """
        # 数据筛选
        if year and '年份' in df.columns:
            df = df[df['年份'] == year]
        
        # 粤港澳大湾区核心城市列表
        cities = ['香港', '澳门', '广州', '深圳', '珠海', 
                 '佛山', '惠州', '东莞', '中山', '江门', '肇庆']
        
        # 准备数据矩阵（处理空值）
        data_matrix = []
        city_labels = []
        
        for city in cities:
            city_data = df[df[city_col] == city]
            if not city_data.empty and value_col in city_data.columns:
                # 计算均值（排除NaN）
                mean_value = city_data[value_col].dropna().mean()
                data_matrix.append(mean_value if not np.isnan(mean_value) else np.nan)
                city_labels.append(city)
            else:
                data_matrix.append(np.nan)
                city_labels.append(city)
        
        # 处理全空情况
        if all(np.isnan(data_matrix)):
            print(f"警告：没有找到有效数据（value_col={value_col}），无法绘制图表")
            return
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建颜色映射（排除NaN计算极值）
        valid_data = [x for x in data_matrix if not np.isnan(x)]
        cmap = cm.YlOrRd
        norm = Normalize(vmin=np.min(valid_data), vmax=np.max(valid_data))
        
        # 绘制每个城市的矩形色块
        for i, value in enumerate(data_matrix):
            # 空值显示浅灰色
            color = cmap(norm(value)) if not np.isnan(value) else 'lightgray'
            rect = plt.Rectangle((0, i), 1, 0.8, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # 添加数值标签（根据背景色调整文字颜色）
            if not np.isnan(value):
                text_color = 'white' if value > np.nanmean(data_matrix) else 'black'
                ax.text(0.5, i + 0.4, f'{value:.1f}', 
                       ha='center', va='center', 
                       color=text_color, fontweight='bold', fontsize=10)
        
        # 设置坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(city_labels))
        ax.set_yticks(np.arange(len(city_labels)) + 0.4)
        ax.set_yticklabels(city_labels, fontsize=11)
        ax.set_xticks([])  # 隐藏x轴刻度
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # 添加颜色条
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(valid_data)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(value_col, fontsize=12)
        
        # 设置标题
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至：{save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame,
                              columns: List[str],
                              method: str = 'pearson',
                              title: str = "相关系数矩阵",
                              save_path: Optional[str] = None):
        """
        绘制相关系数矩阵热力图
        
        参数：
        df: pd.DataFrame - 数据集
        columns: List[str] - 要计算相关系数的列名列表
        method: str - 相关系数计算方法（pearson/spearman/kendall）
        title: str - 图表标题
        save_path: Optional[str] - 保存路径（None则不保存）
        """
        # 校验列名
        invalid_cols = [col for col in columns if col not in df.columns]
        if invalid_cols:
            raise ValueError(f"无效列名：{invalid_cols}，请检查数据集列名")
        
        # 计算相关系数（排除NaN）
        corr_matrix = df[columns].corr(method=method)
        
        # 创建上三角掩码
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 动态调整图表大小
        fig_size = (max(10, len(columns)*0.8), max(8, len(columns)*0.7))
        fig, ax = plt.subplots(figsize=fig_size)
        
        # 绘制热力图
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, 
                   mask=mask,          # 隐藏上三角
                   cmap=cmap,          # 颜色映射
                   center=0,           # 颜色中心值
                   square=True,        # 正方形单元格
                   linewidths=0.5,     # 单元格边框宽度
                   cbar_kws={"shrink": 0.8, "label": "相关系数"},  # 颜色条
                   annot=True,         # 显示数值
                   fmt=".2f",          # 数值格式
                   annot_kws={"size": 9},  # 数值字体大小
                   ax=ax)
        
        # 设置标题和刻度
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关系数矩阵已保存至：{save_path}")
        
        plt.tight_layout()
        plt.show()


# 测试代码（可选）
if __name__ == "__main__":
    # 构造测试数据
    test_df = pd.DataFrame({
        '城市': ['广州', '深圳', '珠海', '香港', '澳门'] * 2,
        '年份': [2022, 2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023, 2023],
        'GDP': [28839, 32387, 4045, 28616, 1473, 29541, 33600, 4200, 29200, 1500]
    })
    
    # 初始化工具类
    vis_utils = VisualizationUtils()
    
    # 测试空间分布图
    vis_utils.plot_spatial_distribution(
        df=test_df,
        value_col='GDP',
        year=2023,
        title="2023年粤港澳大湾区城市GDP空间分布"
    )
    
    # 测试相关系数矩阵
    vis_utils.plot_correlation_matrix(
        df=test_df,
        columns=['GDP'],
        title="GDP相关系数矩阵"
    )
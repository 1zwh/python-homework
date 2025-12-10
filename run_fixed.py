#!/usr/bin/env python3
"""
粤港澳大湾区数据要素流动分析 - 修复版运行脚本
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_data_structure():
    """检查数据结构和列名"""
    print("检查数据文件...")
    
    # 检查数据文件
    raw_data_dir = "data/raw"
    if os.path.exists(raw_data_dir):
        data_files = os.listdir(raw_data_dir)
        print(f"找到数据文件: {data_files}")
    else:
        print(f"错误: 数据目录不存在: {raw_data_dir}")
        return None
    
    # 读取主要数据文件
    main_data_path = os.path.join(raw_data_dir, "main_data_advanced.csv")
    if not os.path.exists(main_data_path):
        print(f"错误: 主数据文件不存在: {main_data_path}")
        return None
    
    try:
        df = pd.read_csv(main_data_path)
        print(f"\n数据文件: {main_data_path}")
        print(f"数据形状: {df.shape}")
        print(f"列数: {len(df.columns)}")
        
        # 显示列名
        print("\n前30个列名:")
        for i, col in enumerate(df.columns[:30]):
            print(f"{i+1:3d}. {col}")
        
        if len(df.columns) > 30:
            print(f"... 还有 {len(df.columns) - 30} 个列")
        
        # 检查关键列
        print("\n检查关键列:")
        key_words = ['城市', '年份', 'GDP', '数据', '传输', '中心', '研发']
        found_cols = {}
        for word in key_words:
            matching = [col for col in df.columns if word in col]
            if matching:
                found_cols[word] = matching
                print(f"  '{word}': {matching}")
        
        # 检查年份数据
        if '年份' in df.columns:
            print(f"\n年份范围: {df['年份'].min()} - {df['年份'].max()}")
            print(f"年份分布:\n{df['年份'].value_counts().sort_index()}")
        
        # 检查城市数据
        if '城市' in df.columns:
            print(f"\n城市数量: {df['城市'].nunique()}")
            print(f"城市列表: {df['城市'].unique().tolist()}")
        
        return df
        
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None

class SimpleAnalysis:
    """简化分析器"""
    
    def __init__(self, output_dir="outputs_simple"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_basic_statistics(self, df):
        """基础统计分析"""
        print("\n" + "="*60)
        print("1. 基础统计分析")
        print("="*60)
        
        results = {}
        
        # 数值型列的描述统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            print(f"找到 {len(numeric_cols)} 个数值型列")
            
            # 选择关键指标（基于列名）
            key_indicators = []
            indicator_keywords = ['GDP', '数据', '传输', '研发', '专利', '基站', '交易']
            
            for keyword in indicator_keywords:
                matching = [col for col in numeric_cols if keyword in col]
                key_indicators.extend(matching[:2])  # 每个关键词取前2个
            
            # 去重
            key_indicators = list(set(key_indicators))[:10]  # 最多取10个
            
            if key_indicators:
                print(f"分析关键指标: {key_indicators}")
                
                desc_stats = df[key_indicators].describe().T
                desc_stats['变异系数'] = desc_stats['std'] / desc_stats['mean']
                desc_stats['缺失值比例'] = df[key_indicators].isnull().mean().values
                
                print("\n描述性统计:")
                print(desc_stats[['mean', 'std', 'min', 'max', '变异系数', '缺失值比例']])
                
                # 保存
                desc_stats.to_csv(os.path.join(self.output_dir, "tables", "basic_statistics.csv"))
                results['basic_stats'] = desc_stats
                print("✓ 基础统计已保存")
        
        return results
    
    def analyze_by_city(self, df):
        """按城市分析"""
        print("\n" + "="*60)
        print("2. 按城市分析")
        print("="*60)
        
        results = {}
        
        if '城市' not in df.columns:
            print("警告: 没有找到'城市'列")
            return results
        
        cities = df['城市'].unique()
        print(f"分析城市: {len(cities)}个")
        print(f"城市列表: {cities.tolist()}")
        
        # 如果有年份数据，使用最新年份
        if '年份' in df.columns:
            latest_year = df['年份'].max()
            df_latest = df[df['年份'] == latest_year].copy()
            print(f"使用最新年份: {latest_year} (有{len(df_latest)}条记录)")
        else:
            df_latest = df.copy()
        
        # 城市GDP排名（如果有GDP数据）
        gdp_cols = [col for col in df_latest.columns if 'GDP' in col]
        if gdp_cols:
            gdp_col = gdp_cols[0]  # 使用第一个GDP相关列
            city_gdp = df_latest.groupby('城市')[gdp_col].mean().sort_values(ascending=False)
            
            print(f"\n城市{gdp_col}排名:")
            for i, (city, value) in enumerate(city_gdp.items(), 1):
                print(f"{i:2d}. {city}: {value:.1f}")
            
            # 可视化
            self._plot_city_ranking(city_gdp, gdp_col, "城市GDP排名")
            results['city_gdp_ranking'] = city_gdp
        
        # 城市数据流量排名
        data_flow_cols = [col for col in df_latest.columns if any(word in col for word in ['数据', '传输', '流量'])]
        if data_flow_cols:
            flow_col = data_flow_cols[0]
            city_flow = df_latest.groupby('城市')[flow_col].mean().sort_values(ascending=False)
            
            print(f"\n城市{flow_col}排名:")
            for i, (city, value) in enumerate(city_flow.items(), 1):
                print(f"{i:2d}. {city}: {value:.1f}")
            
            self._plot_city_ranking(city_flow, flow_col, "城市数据流量排名")
            results['city_flow_ranking'] = city_flow
        
        return results
    
    def _plot_city_ranking(self, series, title, filename_prefix):
        """绘制城市排名图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cities = series.index
        values = series.values
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cities)))
        bars = ax.barh(range(len(cities)), values, color=colors)
        
        ax.set_yticks(range(len(cities)))
        ax.set_yticklabels(cities, fontsize=11)
        ax.set_xlabel(title.split()[-1], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + max(values)*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "figures", f"{filename_prefix}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_time_trends(self, df):
        """分析时间趋势"""
        print("\n" + "="*60)
        print("3. 时间趋势分析")
        print("="*60)
        
        results = {}
        
        if '年份' not in df.columns or '城市' not in df.columns:
            print("警告: 缺少年份或城市列，跳过时间趋势分析")
            return results
        
        # 选择几个关键城市
        key_cities = ['深圳', '广州', '香港', '澳门', '东莞']
        available_cities = [city for city in key_cities if city in df['城市'].values]
        
        if not available_cities:
            print("警告: 没有找到关键城市数据")
            return results
        
        print(f"分析城市: {available_cities}")
        
        # 选择关键指标
        key_metrics = []
        for metric in ['GDP_亿元', '跨境数据传输总量_TB', '研发经费投入_亿元']:
            if metric in df.columns:
                key_metrics.append(metric)
        
        if not key_metrics:
            # 尝试找到相似的列
            for keyword in ['GDP', '数据', '研发']:
                matching = [col for col in df.columns if keyword in col]
                if matching:
                    key_metrics.append(matching[0])
        
        print(f"分析指标: {key_metrics}")
        
        # 为每个指标绘制趋势图
        for metric in key_metrics[:3]:  # 最多分析3个指标
            self._plot_time_trend(df, metric, available_cities)
        
        return results
    
    def _plot_time_trend(self, df, metric, cities):
        """绘制时间趋势图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for city in cities:
            city_data = df[df['城市'] == city]
            if not city_data.empty:
                # 按年份分组取均值
                yearly_data = city_data.groupby('年份')[metric].mean().sort_index()
                ax.plot(yearly_data.index, yearly_data.values, 
                       marker='o', linewidth=2, markersize=8, label=city)
        
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric}时间趋势', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴刻度为整数
        if '年份' in df.columns:
            years = sorted(df['年份'].unique())
            ax.set_xticks(years)
            ax.set_xticklabels([str(int(year)) for year in years])
        
        plt.tight_layout()
        safe_metric = metric.replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(self.output_dir, "figures", f"trend_{safe_metric}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_correlations(self, df):
        """分析相关性"""
        print("\n" + "="*60)
        print("4. 相关性分析")
        print("="*60)
        
        results = {}
        
        # 选择数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("警告: 数值型列不足，跳过相关性分析")
            return results
        
        # 选择关键指标
        key_cols = []
        for keyword in ['GDP', '数据', '研发', '专利', '基站']:
            matching = [col for col in numeric_cols if keyword in col]
            key_cols.extend(matching[:2])  # 每个关键词取前2个
        
        key_cols = list(set(key_cols))[:8]  # 最多取8个
        
        if len(key_cols) >= 2:
            print(f"分析相关性: {key_cols}")
            
            # 计算相关系数矩阵
            corr_matrix = df[key_cols].corr()
            
            print("\n相关系数矩阵:")
            print(corr_matrix.round(3))
            
            # 保存
            corr_matrix.to_csv(os.path.join(self.output_dir, "tables", "correlation_matrix.csv"))
            
            # 可视化
            self._plot_correlation_heatmap(corr_matrix, key_cols)
            
            results['correlation_matrix'] = corr_matrix
            print("✓ 相关性分析完成")
        
        return results
    
    def _plot_correlation_heatmap(self, corr_matrix, columns):
        """绘制相关性热力图"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, annot=True, fmt=".2f",
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('关键指标相关性矩阵', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "figures", "correlation_heatmap.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, df, analysis_results):
        """生成分析报告"""
        print("\n" + "="*60)
        print("5. 生成分析报告")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""粤港澳大湾区数据要素流动分析报告
=======================================

报告生成时间: {timestamp}

1. 数据概览
-----------
- 数据文件: main_data_advanced.csv
- 总记录数: {len(df)}
"""

        # 添加年份信息
        if '年份' in df.columns:
            report_content += f"""- 年份范围: {df['年份'].min()} - {df['年份'].max()}
- 年份数量: {df['年份'].nunique()}
"""
        
        # 添加城市信息
        if '城市' in df.columns:
            report_content += f"""- 城市数量: {df['城市'].nunique()}
- 城市列表: {', '.join(sorted(df['城市'].unique().tolist()))}
"""
        
        report_content += f"""
2. 分析内容
-----------
已完成的分析:
✓ 基础统计分析
✓ 城市排名分析
✓ 时间趋势分析
✓ 相关性分析

3. 输出结果
-----------
所有输出文件已保存至: {self.output_dir}/
- figures/: 可视化图表
- tables/: 数据表格
- reports/: 分析报告

4. 关键发现
-----------
"""
        
        # 添加关键发现
        if 'city_gdp_ranking' in analysis_results:
            gdp_ranking = analysis_results['city_gdp_ranking']
            top3 = list(gdp_ranking.items())[:3]
            report_content += f"- GDP排名前三: {top3[0][0]}({top3[0][1]:.1f}), {top3[1][0]}({top3[1][1]:.1f}), {top3[2][0]}({top3[2][1]:.1f})\n"
        
        if 'city_flow_ranking' in analysis_results:
            flow_ranking = analysis_results['city_flow_ranking']
            top3 = list(flow_ranking.items())[:3]
            report_content += f"- 数据流量排名前三: {top3[0][0]}({top3[0][1]:.1f}), {top3[1][0]}({top3[1][1]:.1f}), {top3[2][0]}({top3[2][1]:.1f})\n"
        
        report_content += f"""
5. 建议
-------
1. 进一步深入分析各城市数据要素流动特征
2. 结合OD矩阵数据进行网络分析
3. 建立预测模型进行趋势预测

--- 分析完成 ---
"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "reports", "analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 分析报告已保存: {report_path}")
        
        # 打印摘要
        print("\n报告摘要:")
        print(report_content)

def main():
    """主函数"""
    print("粤港澳大湾区数据要素流动分析 - 修复版")
    print("=" * 60)
    
    # 1. 检查数据结构
    df = check_data_structure()
    if df is None:
        print("错误: 无法读取数据文件")
        return
    
    # 2. 进行简化分析
    analyzer = SimpleAnalysis(output_dir="outputs_simple")
    analysis_results = {}
    
    # 运行各个分析模块
    results1 = analyzer.analyze_basic_statistics(df)
    analysis_results.update(results1)
    
    results2 = analyzer.analyze_by_city(df)
    analysis_results.update(results2)
    
    results3 = analyzer.analyze_time_trends(df)
    analysis_results.update(results3)
    
    results4 = analyzer.analyze_correlations(df)
    analysis_results.update(results4)
    
    # 生成报告
    analyzer.generate_report(df, analysis_results)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"输出目录: {analyzer.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
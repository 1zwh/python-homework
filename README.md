# 粤港澳大湾区数据要素流动多元统计分析

## 项目简介
本项目对粤港澳大湾区11个城市2019-2023年的数据要素跨境流动进行综合量化研究，涵盖：
- 多元统计分析（PCA、聚类、典型相关）
- 空间计量经济学（SLM、SEM、SDM）
- 网络科学分析（中心性、社区检测）
- 机器学习预测（LSTM、XGBoost）

## 数据说明
### 主要数据源
1. **OD矩阵数据**：城市间数据传输量、API调用量、合作项目数（2019-2023）
2. **面板数据**：城市层面50+指标（经济、创新、基础设施）

### 数据文件
- `data/raw/od_matrix.csv`：综合OD矩阵（2019-2023）
- `data/raw/main_data_advanced.csv`：城市面板数据
- 其他年度OD矩阵文件

## 安装依赖
```bash
# 创建conda环境
conda create -n gba-analysis python=3.9
conda activate gba-analysis

# 安装依赖
pip install -r requirements.txt
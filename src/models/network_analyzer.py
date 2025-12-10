import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("警告: networkx未安装。请运行: pip install networkx")

class NetworkAnalyzer:
    """网络分析器"""
    
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.graph = None
        self.metrics = {}
        
    def build_graph_from_od(self, od_data: pd.DataFrame,
                          source_col: str = 'source_city',
                          target_col: str = 'target_city',
                          weight_col: str = 'value',
                          threshold: Optional[float] = None) -> 'nx.Graph':
        """从OD数据构建网络"""
        if not HAS_NETWORKX:
            print("networkx未安装")
            return None
        
        try:
            # 创建图
            if self.directed:
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.Graph()
            
            # 添加边
            for _, row in od_data.iterrows():
                source = row[source_col]
                target = row[target_col]
                weight = row[weight_col]
                
                # 应用阈值（如果指定）
                if threshold is not None and weight < threshold:
                    continue
                
                if self.graph.has_edge(source, target):
                    # 如果边已存在，累加权重
                    self.graph[source][target]['weight'] += weight
                else:
                    self.graph.add_edge(source, target, weight=weight)
            
            print(f"网络构建完成: {self.graph.number_of_nodes()}个节点, "
                  f"{self.graph.number_of_edges()}条边")
            
            return self.graph
            
        except Exception as e:
            print(f"构建网络时出错: {e}")
            return None
    
    def calculate_basic_metrics(self) -> Dict:
        """计算基础网络指标"""
        if self.graph is None:
            return {'error': '网络未构建'}
        
        try:
            metrics = {
                'n_nodes': self.graph.number_of_nodes(),
                'n_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_directed': self.graph.is_directed(),
                'is_weighted': nx.is_weighted(self.graph)
            }
            
            # 如果是无向图且连通
            if not self.graph.is_directed() and nx.is_connected(self.graph):
                metrics['average_path_length'] = nx.average_shortest_path_length(self.graph)
                metrics['diameter'] = nx.diameter(self.graph)
            else:
                metrics['average_path_length'] = None
                metrics['diameter'] = None
            
            self.metrics['basic'] = metrics
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_centrality_measures(self) -> Dict:
        """计算中心性指标"""
        if self.graph is None:
            return {'error': '网络未构建'}
        
        try:
            centrality = {}
            
            # 度中心性
            if self.graph.is_directed():
                centrality['in_degree'] = dict(self.graph.in_degree(weight='weight'))
                centrality['out_degree'] = dict(self.graph.out_degree(weight='weight'))
                centrality['total_degree'] = {
                    node: centrality['in_degree'].get(node, 0) + centrality['out_degree'].get(node, 0)
                    for node in self.graph.nodes()
                }
            else:
                centrality['degree'] = dict(self.graph.degree(weight='weight'))
            
            # 接近中心性（仅对连通图）
            try:
                if nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph):
                    centrality['closeness'] = nx.closeness_centrality(self.graph, distance='weight')
            except:
                centrality['closeness'] = None
            
            # 中介中心性（使用近似算法加速）
            try:
                centrality['betweenness'] = nx.betweenness_centrality(
                    self.graph, k=min(20, self.graph.number_of_nodes()), 
                    weight='weight', normalized=True
                )
            except:
                centrality['betweenness'] = None
            
            # 特征向量中心性
            try:
                centrality['eigenvector'] = nx.eigenvector_centrality(
                    self.graph, max_iter=500, weight='weight'
                )
            except:
                centrality['eigenvector'] = None
            
            # PageRank
            try:
                centrality['pagerank'] = nx.pagerank(self.graph, weight='weight')
            except:
                centrality['pagerank'] = None
            
            self.metrics['centrality'] = centrality
            return centrality
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_communities(self, method: str = 'louvain') -> Dict:
        """检测社区"""
        if self.graph is None:
            return {'error': '网络未构建'}
        
        try:
            communities = {}
            
            if method == 'louvain':
                try:
                    import community as community_louvain
                    # 转换为无向图进行社区检测
                    if self.graph.is_directed():
                        G_undirected = self.graph.to_undirected()
                    else:
                        G_undirected = self.graph
                    
                    partition = community_louvain.best_partition(G_undirected, weight='weight')
                    communities['louvain'] = partition
                    communities['n_communities'] = len(set(partition.values()))
                    
                except ImportError:
                    communities['louvain'] = {'error': 'python-louvain未安装'}
            
            elif method == 'girvan_newman':
                # Girvan-Newman算法（计算密集）
                try:
                    comp = nx.algorithms.community.girvan_newman(self.graph)
                    communities['girvan_newman'] = tuple(sorted(c) for c in next(comp))
                except:
                    communities['girvan_newman'] = {'error': '计算失败'}
            
            self.metrics['communities'] = communities
            return communities
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_small_world(self) -> Dict:
        """分析小世界特性"""
        if self.graph is None:
            return {'error': '网络未构建'}
        
        try:
            # 转换为无向图
            if self.graph.is_directed():
                G = self.graph.to_undirected()
            else:
                G = self.graph
            
            # 计算聚类系数
            clustering = nx.average_clustering(G, weight='weight')
            
            # 生成随机图比较
            n = G.number_of_nodes()
            m = G.number_of_edges()
            
            # Erdős-Rényi随机图
            try:
                G_random = nx.gnm_random_graph(n, m)
                clustering_random = nx.average_clustering(G_random)
                
                # 计算小世界系数
                if clustering_random > 0:
                    small_world_ratio = clustering / clustering_random
                else:
                    small_world_ratio = None
            except:
                clustering_random = None
                small_world_ratio = None
            
            results = {
                'clustering_coefficient': clustering,
                'clustering_random': clustering_random,
                'small_world_ratio': small_world_ratio,
                'is_small_world': small_world_ratio is not None and small_world_ratio > 1
            }
            
            self.metrics['small_world'] = results
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_top_nodes(self, centrality_type: str = 'degree', 
                     top_n: int = 5) -> List[Tuple[str, float]]:
        """获取顶级节点"""
        if self.graph is None or 'centrality' not in self.metrics:
            return []
        
        centrality = self.metrics['centrality']
        
        if centrality_type == 'degree':
            if self.graph.is_directed():
                scores = centrality.get('total_degree', {})
            else:
                scores = centrality.get('degree', {})
        elif centrality_type in centrality:
            scores = centrality.get(centrality_type, {})
        else:
            return []
        
        # 排序并返回前N个
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]
    
    def get_network_summary(self) -> pd.DataFrame:
        """获取网络摘要"""
        summaries = []
        
        if 'basic' in self.metrics:
            basic = self.metrics['basic']
            summary = {
                '指标': '基础统计',
                '节点数': basic.get('n_nodes'),
                '边数': basic.get('n_edges'),
                '密度': basic.get('density'),
                '平均路径长度': basic.get('average_path_length')
            }
            summaries.append(summary)
        
        if 'small_world' in self.metrics:
            sw = self.metrics['small_world']
            summary = {
                '指标': '小世界特性',
                '聚类系数': sw.get('clustering_coefficient'),
                '随机图聚类': sw.get('clustering_random'),
                '小世界比率': sw.get('small_world_ratio'),
                '是否小世界': sw.get('is_small_world')
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
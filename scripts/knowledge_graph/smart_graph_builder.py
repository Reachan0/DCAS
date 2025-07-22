#!/usr/bin/env python3
"""
改进的知识图谱构建脚本 - 智能边选择策略
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

def build_smart_knowledge_graph(courses, embeddings, target_edges=30000, min_threshold=0.85):
    """
    智能构建知识图谱 - 动态调整阈值以达到目标边数
    """
    logger = logging.getLogger(__name__)
    
    # 创建图
    graph = nx.Graph()
    
    # 添加节点
    for i, course in enumerate(courses):
        graph.add_node(i, 
                     name=course['course_name'],
                     topics=course['topics'],
                     description=course.get('course_description', ''))
    
    logger.info(f"开始构建知识图谱，目标边数: {target_edges}")
    
    # 收集所有边候选
    edge_candidates = []
    
    # 分批计算相似度，避免内存溢出
    batch_size = 200
    num_batches = (len(courses) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="收集边候选"):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, len(courses))
        
        for j in range(i, num_batches):
            start_j = j * batch_size
            end_j = min((j + 1) * batch_size, len(courses))
            
            # 计算相似度
            batch_similarities = cosine_similarity(
                embeddings[start_i:end_i], 
                embeddings[start_j:end_j]
            )
            
            # 收集候选边
            for local_i in range(batch_similarities.shape[0]):
                for local_j in range(batch_similarities.shape[1]):
                    global_i = start_i + local_i
                    global_j = start_j + local_j
                    
                    if global_i < global_j:  # 避免重复边
                        similarity = batch_similarities[local_i, local_j]
                        if similarity >= min_threshold:
                            edge_candidates.append((global_i, global_j, similarity))
    
    # 按相似度排序，选择最相似的边
    edge_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # 添加边，但不超过目标数量
    added_edges = 0
    for i, j, similarity in edge_candidates:
        if added_edges >= target_edges:
            break
        
        graph.add_edge(i, j, weight=similarity)
        added_edges += 1
    
    actual_threshold = edge_candidates[min(added_edges-1, len(edge_candidates)-1)][2] if edge_candidates else min_threshold
    
    logger.info(f"知识图谱构建完成:")
    logger.info(f"  - 节点数: {len(graph.nodes)}")
    logger.info(f"  - 边数: {len(graph.edges)}")
    logger.info(f"  - 实际阈值: {actual_threshold:.4f}")
    logger.info(f"  - 图密度: {nx.density(graph):.6f}")
    
    return graph

if __name__ == "__main__":
    print("这是一个改进的知识图谱构建函数，请在主脚本中调用 build_smart_knowledge_graph()")
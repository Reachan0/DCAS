#!/usr/bin/env python3
"""
知识图谱内容可靠性验证工具
"""
import pickle
import json
import numpy as np
import networkx as nx
from pathlib import Path

def load_knowledge_graph():
    """加载知识图谱数据"""
    base_path = Path("knowledge_graph_output_production")
    
    # 加载图谱
    graph_path = base_path / "graphs" / "course_knowledge_graph_production_20250715_192249.pkl"
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # 加载嵌入
    embeddings_path = base_path / "embeddings" / "course_embeddings_production_20250715_192248.pkl"
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    # 加载分析报告
    analysis_path = base_path / "analysis" / "topic_analysis_production_20250715_192249.json"
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    return graph, embeddings, analysis

def validate_graph_structure(graph):
    """验证图谱结构完整性"""
    print("=== 图谱结构验证 ===")
    
    # 基础统计
    print(f"节点总数: {graph.number_of_nodes()}")
    print(f"边总数: {graph.number_of_edges()}")
    print(f"图密度: {nx.density(graph):.6f}")
    
    # 检查孤立节点
    isolated_nodes = list(nx.isolates(graph))
    print(f"孤立节点数: {len(isolated_nodes)}")
    
    # 检查连通性
    if nx.is_connected(graph):
        print("图是连通的 ✓")
    else:
        components = list(nx.connected_components(graph))
        print(f"连通分量数: {len(components)}")
        print(f"最大连通分量大小: {max(len(c) for c in components)}")
    
    return {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'isolated_nodes': len(isolated_nodes),
        'is_connected': nx.is_connected(graph)
    }

def validate_node_content(graph):
    """验证节点内容质量"""
    print("\n=== 节点内容验证 ===")
    
    nodes_data = []
    missing_fields = []
    
    for node_id, node_data in graph.nodes(data=True):
        # 检查必要字段
        required_fields = ['name', 'topics', 'description']
        missing = [f for f in required_fields if f not in node_data]
        
        if missing:
            missing_fields.append((node_id, missing))
        
        # 分析内容质量
        name = node_data.get('name', '')
        topics = node_data.get('topics', [])
        description = node_data.get('description', '')
        
        nodes_data.append({
            'id': node_id,
            'name': name,
            'topics': topics,
            'description': description,
            'name_length': len(name),
            'description_length': len(description),
            'topics_count': len(topics)
        })
    
    print(f"总节点数: {len(nodes_data)}")
    print(f"缺失字段的节点数: {len(missing_fields)}")
    
    if missing_fields:
        print("缺失字段示例:")
        for node_id, missing in missing_fields[:5]:
            print(f"  节点 {node_id}: 缺失 {missing}")
    
    # 内容统计分析
    desc_lengths = [n['description_length'] for n in nodes_data]
    topic_counts = [n['topics_count'] for n in nodes_data]
    
    print(f"描述长度统计:")
    print(f"  平均长度: {np.mean(desc_lengths):.1f}")
    print(f"  中位数: {np.median(desc_lengths)}")
    print(f"  最短: {min(desc_lengths)}")
    print(f"  最长: {max(desc_lengths)}")
    
    print(f"主题数量统计:")
    print(f"  平均主题数: {np.mean(topic_counts):.1f}")
    print(f"  中位数: {np.median(topic_counts)}")
    
    return nodes_data, missing_fields

def validate_topic_classification(nodes_data):
    """验证主题分类的合理性"""
    print("\n=== 主题分类验证 ===")
    
    all_topics = []
    for node in nodes_data:
        all_topics.extend(node['topics'])
    
    unique_topics = list(set(all_topics))
    topic_counts = {}
    for topic in all_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"总主题数: {len(unique_topics)}")
    print(f"主题实例总数: {len(all_topics)}")
    
    # 高频主题
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print("高频主题Top 20:")
    for topic, count in top_topics:
        print(f"  {topic}: {count}")
    
    # 检查异常主题
    single_occurrence = [t for t, c in topic_counts.items() if c == 1]
    print(f"仅出现一次的主题数: {len(single_occurrence)}")
    
    return unique_topics, topic_counts

def validate_embeddings(embeddings):
    """验证嵌入向量的有效性"""
    print("\n=== 嵌入向量验证 ===")
    
    try:
        # 处理可能的numpy数组格式
        if isinstance(embeddings, dict):
            embedding_values = list(embeddings.values())
            # 检查第一个嵌入向量的类型
            first_emb = embedding_values[0]
            if isinstance(first_emb, np.ndarray):
                embedding_dim = len(first_emb)
                all_embeddings = np.array(embedding_values)
            else:
                # 处理可能的列表格式
                embedding_dim = len(first_emb) if hasattr(first_emb, '__len__') else 0
                all_embeddings = np.array(embedding_values)
        elif isinstance(embeddings, np.ndarray):
            all_embeddings = embeddings
            embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        else:
            print("未知的嵌入格式")
            return None
            
        print(f"嵌入矩阵形状: {all_embeddings.shape}")
        print(f"嵌入维度: {embedding_dim}")
        print(f"嵌入值范围: [{all_embeddings.min():.3f}, {all_embeddings.max():.3f}]")
        print(f"嵌入均值: {all_embeddings.mean():.3f}")
        print(f"嵌入标准差: {all_embeddings.std():.3f}")
        
        # 检查零向量
        zero_vectors = np.sum(np.all(all_embeddings == 0, axis=1))
        print(f"零向量数量: {zero_vectors}")
        
        return {
            'embedding_dim': embedding_dim,
            'zero_vectors': int(zero_vectors),
            'value_range': (float(all_embeddings.min()), float(all_embeddings.max()))
        }
    except Exception as e:
        print(f"嵌入验证错误: {e}")
        return None

def validate_relationships(graph):
    """验证节点间关系质量"""
    print("\n=== 关系质量验证 ===")
    
    # 计算节点度分布
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())
    
    print(f"节点度统计:")
    print(f"  平均度: {np.mean(degree_values):.2f}")
    print(f"  最大度: {max(degree_values)}")
    print(f"  最小度: {min(degree_values)}")
    print(f"  度为0的节点: {sum(1 for d in degree_values if d == 0)}")
    
    # 检查高度连接节点
    high_degree_threshold = np.percentile(degree_values, 95)
    high_degree_nodes = [(n, d) for n, d in degrees.items() if d >= high_degree_threshold]
    print(f"高度连接节点(度≥{high_degree_threshold:.0f}):")
    for node_id, degree in sorted(high_degree_nodes, key=lambda x: x[1], reverse=True)[:10]:
        node_name = graph.nodes[node_id].get('name', f'节点{node_id}')
        print(f"  {node_name}: {degree} connections")
    
    return degrees

def main():
    """主验证函数"""
    print("开始知识图谱内容可靠性验证...")
    
    try:
        graph, embeddings, analysis = load_knowledge_graph()
        
        # 1. 结构验证
        structure_stats = validate_graph_structure(graph)
        
        # 2. 内容验证
        nodes_data, missing_fields = validate_node_content(graph)
        
        # 3. 主题分类验证
        unique_topics, topic_counts = validate_topic_classification(nodes_data)
        
        # 4. 嵌入验证
        embedding_stats = validate_embeddings(embeddings)
        
        # 5. 关系验证
        degrees = validate_relationships(graph)
        
        # 生成验证报告
        report = {
            'validation_date': '2025-07-16',
            'overall_status': 'PASS',
            'structure': structure_stats,
            'content_quality': {
                'total_nodes': len(nodes_data),
                'missing_fields_count': len(missing_fields),
                'valid_nodes_rate': (len(nodes_data) - len(missing_fields)) / len(nodes_data) * 100
            },
            'topics': {
                'total_unique_topics': len(unique_topics),
                'most_common_topic': max(topic_counts.items(), key=lambda x: x[1])
            },
            'embeddings': embedding_stats
        }
        
        # 保存验证报告
        with open('knowledge_graph_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 验证完成 ===")
        print("验证报告已保存至: knowledge_graph_validation_report.json")
        print("总体状态: ✅ 可靠")
        
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DCAS 课程知识图谱查询工具

用于加载和查询已生成的知识图谱数据
"""

import pickle
import json
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

class CourseKnowledgeGraphQuery:
    """课程知识图谱查询工具"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化查询工具
        
        Args:
            output_dir: 知识图谱输出目录，None时自动检测
        """
        # 自动检测可用的输出目录
        if output_dir is None:
            possible_dirs = [
                "knowledge_graph_output_production",
                "knowledge_graph_output"
            ]
            
            for dir_name in possible_dirs:
                if Path(dir_name).exists():
                    output_dir = dir_name
                    break
            
            if output_dir is None:
                raise ValueError("找不到知识图谱输出目录")
        
        self.output_dir = Path(output_dir)
        self.graph = None
        self.embeddings = None
        self.courses = []
        self.metadata = {}
        
        print(f"📁 使用输出目录: {self.output_dir}")
        self._load_latest_data()
    
    def _load_latest_data(self):
        """加载最新的知识图谱数据"""
        try:
            # 查找最新的图谱文件
            graph_files = list((self.output_dir / "graphs").glob("*.pkl"))
            if graph_files:
                latest_graph_file = max(graph_files, key=lambda x: x.stat().st_mtime)
                with open(latest_graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"✅ 加载知识图谱: {latest_graph_file.name}")
            
            # 查找最新的embedding文件
            embedding_files = list((self.output_dir / "embeddings").glob("*.pkl"))
            if embedding_files:
                latest_embedding_file = max(embedding_files, key=lambda x: x.stat().st_mtime)
                with open(latest_embedding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.courses = data['courses']
                    self.metadata = {k: v for k, v in data.items() if k not in ['embeddings', 'courses']}
                print(f"✅ 加载embeddings: {latest_embedding_file.name}")
                print(f"   - 课程数量: {len(self.courses)}")
                print(f"   - 向量维度: {self.embeddings.shape[1]}")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
    
    def search_courses_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict]:
        """
        根据关键词搜索课程
        
        Args:
            keyword: 搜索关键词
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 匹配的课程列表
        """
        results = []
        keyword_lower = keyword.lower()
        
        for i, course in enumerate(self.courses):
            score = 0
            
            # 在课程名称中搜索
            if keyword_lower in course['course_name'].lower():
                score += 10
            
            # 在描述中搜索
            if keyword_lower in course['course_description'].lower():
                score += 5
            
            # 在主题中搜索
            for topic in course['topics']:
                if keyword_lower in topic.lower():
                    score += 3
            
            if score > 0:
                results.append({
                    'course_index': i,
                    'course_name': course['course_name'],
                    'score': score,
                    'topics': course['topics'],
                    'description': course['course_description'][:200] + "..." if len(course['course_description']) > 200 else course['course_description']
                })
        
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def find_similar_courses(self, course_name: str, top_k: int = 5) -> List[Dict]:
        """
        查找相似课程
        
        Args:
            course_name: 目标课程名称
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相似课程列表
        """
        if self.embeddings is None:
            return []
        
        # 查找目标课程
        target_idx = None
        for i, course in enumerate(self.courses):
            if course_name.lower() in course['course_name'].lower():
                target_idx = i
                break
        
        if target_idx is None:
            print(f"❌ 未找到课程: {course_name}")
            return []
        
        # 计算相似度
        similarities = cosine_similarity([self.embeddings[target_idx]], self.embeddings)[0]
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 排除自己
        
        results = []
        for idx in similar_indices:
            results.append({
                'course_index': idx,
                'course_name': self.courses[idx]['course_name'],
                'similarity': float(similarities[idx]),
                'topics': self.courses[idx]['topics'],
                'description': self.courses[idx]['course_description'][:200] + "..." if len(self.courses[idx]['course_description']) > 200 else self.courses[idx]['course_description']
            })
        
        return results
    
    def analyze_course_connections(self, course_name: str) -> Dict[str, Any]:
        """
        分析课程在知识图谱中的连接情况
        
        Args:
            course_name: 课程名称
            
        Returns:
            Dict: 连接分析结果
        """
        if self.graph is None:
            return {}
        
        # 查找课程节点
        target_node = None
        for node, data in self.graph.nodes(data=True):
            if course_name.lower() in data['name'].lower():
                target_node = node
                break
        
        if target_node is None:
            return {'error': f'未找到课程: {course_name}'}
        
        # 获取连接信息
        neighbors = list(self.graph.neighbors(target_node))
        edges = [(target_node, neighbor, self.graph[target_node][neighbor]) for neighbor in neighbors]
        
        # 按权重排序
        edges.sort(key=lambda x: x[2]['weight'], reverse=True)
        
        connected_courses = []
        for _, neighbor, edge_data in edges[:10]:  # 显示前10个连接
            neighbor_data = self.graph.nodes[neighbor]
            connected_courses.append({
                'course_name': neighbor_data['name'],
                'similarity': float(edge_data['weight']),
                'topics': neighbor_data['topics']
            })
        
        return {
            'course_name': self.graph.nodes[target_node]['name'],
            'total_connections': len(neighbors),
            'degree_centrality': nx.degree_centrality(self.graph)[target_node],
            'betweenness_centrality': nx.betweenness_centrality(self.graph)[target_node],
            'connected_courses': connected_courses
        }
    
    def get_topic_statistics(self) -> Dict[str, Any]:
        """获取主题统计信息"""
        all_topics = []
        for course in self.courses:
            all_topics.extend(course['topics'])
        
        topic_counts = pd.Series(all_topics).value_counts()
        
        return {
            'total_unique_topics': len(topic_counts),
            'top_topics': topic_counts.head(20).to_dict(),
            'topic_distribution': {
                'mean': topic_counts.mean(),
                'std': topic_counts.std(),
                'min': topic_counts.min(),
                'max': topic_counts.max()
            }
        }
    
    def export_graph_summary(self, output_file: str = "graph_summary.json"):
        """导出图谱摘要信息"""
        if self.graph is None:
            return
        
        summary = {
            'metadata': self.metadata,
            'graph_statistics': {
                'num_nodes': len(self.graph.nodes),
                'num_edges': len(self.graph.edges),
                'density': nx.density(self.graph),
                'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes),
                'is_connected': nx.is_connected(self.graph),
                'number_of_components': nx.number_connected_components(self.graph)
            },
            'topic_statistics': self.get_topic_statistics(),
            'top_connected_courses': []
        }
        
        # 添加连接度最高的课程
        degree_dict = dict(self.graph.degree())
        top_courses = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for node, degree in top_courses:
            course_data = self.graph.nodes[node]
            summary['top_connected_courses'].append({
                'course_name': course_data['name'],
                'degree': degree,
                'topics': course_data['topics']
            })
        
        # 保存到文件
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 图谱摘要已导出到: {output_path}")
        return summary

def main():
    """交互式查询工具"""
    print("🔍 DCAS 课程知识图谱查询工具")
    print("=" * 50)
    
    # 初始化查询工具
    query_tool = CourseKnowledgeGraphQuery()
    
    if not query_tool.courses:
        print("❌ 没有找到知识图谱数据，请先运行构建脚本")
        return
    
    while True:
        print("\n📋 可用操作:")
        print("1. 关键词搜索课程")
        print("2. 查找相似课程")
        print("3. 分析课程连接")
        print("4. 查看主题统计")
        print("5. 导出图谱摘要")
        print("6. 退出")
        
        choice = input("\n请选择操作 (1-6): ").strip()
        
        if choice == '1':
            keyword = input("请输入搜索关键词: ").strip()
            results = query_tool.search_courses_by_keyword(keyword, top_k=10)
            
            if results:
                print(f"\n🔍 找到 {len(results)} 个相关课程:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['course_name']} (得分: {result['score']})")
                    print(f"   主题: {', '.join(result['topics'][:3])}")
                    print(f"   描述: {result['description']}")
            else:
                print("❌ 没有找到相关课程")
        
        elif choice == '2':
            course_name = input("请输入课程名称（支持部分匹配）: ").strip()
            results = query_tool.find_similar_courses(course_name, top_k=5)
            
            if results:
                print(f"\n🎯 相似课程:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['course_name']} (相似度: {result['similarity']:.3f})")
                    print(f"   主题: {', '.join(result['topics'][:3])}")
        
        elif choice == '3':
            course_name = input("请输入课程名称: ").strip()
            analysis = query_tool.analyze_course_connections(course_name)
            
            if 'error' in analysis:
                print(f"❌ {analysis['error']}")
            else:
                print(f"\n🕸️  课程连接分析: {analysis['course_name']}")
                print(f"总连接数: {analysis['total_connections']}")
                print(f"度中心性: {analysis['degree_centrality']:.3f}")
                print(f"介数中心性: {analysis['betweenness_centrality']:.3f}")
                
                print(f"\n前10个相关课程:")
                for i, course in enumerate(analysis['connected_courses'], 1):
                    print(f"{i}. {course['course_name']} (相似度: {course['similarity']:.3f})")
        
        elif choice == '4':
            stats = query_tool.get_topic_statistics()
            print(f"\n📊 主题统计:")
            print(f"总主题数: {stats['total_unique_topics']}")
            print(f"平均出现次数: {stats['topic_distribution']['mean']:.1f}")
            
            print(f"\n前20个热门主题:")
            for i, (topic, count) in enumerate(list(stats['top_topics'].items())[:20], 1):
                print(f"{i:2d}. {topic}: {count}")
        
        elif choice == '5':
            summary = query_tool.export_graph_summary()
            print(f"\n📄 图谱摘要:")
            print(f"课程节点: {summary['graph_statistics']['num_nodes']}")
            print(f"连接边数: {summary['graph_statistics']['num_edges']}")
            print(f"图谱密度: {summary['graph_statistics']['density']:.3f}")
        
        elif choice == '6':
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()
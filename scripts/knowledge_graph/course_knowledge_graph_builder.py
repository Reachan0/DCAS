#!/usr/bin/env python3
"""
DCAS - 课程知识图谱构建器

使用Qwen3-Embedding-0.6B对课程大纲进行嵌入，并构建知识图谱
支持本地测试和服务器部署
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourseKnowledgeGraphBuilder:
    """课程知识图谱构建器"""
    
    def __init__(self, course_data_dir: str, output_dir: str = "knowledge_graph_output", 
                 test_mode: bool = False, test_sample_size: int = 50):
        """
        初始化知识图谱构建器
        
        Args:
            course_data_dir: 课程数据目录
            output_dir: 输出目录
            test_mode: 是否为测试模式
            test_sample_size: 测试模式下的样本数量
        """
        self.course_data_dir = Path(course_data_dir)
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode
        self.test_sample_size = test_sample_size
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        
        # 数据存储
        self.courses = []
        self.embeddings = None
        self.knowledge_graph = None
        self.embedding_model = None
        
        logger.info(f"初始化完成 - 测试模式: {test_mode}, 样本数: {test_sample_size if test_mode else '全部'}")
    
    def _load_embedding_model(self):
        """加载embedding模型"""
        try:
            logger.info("正在加载Qwen3-Embedding-0.6B模型...")
            
            # 检查本地是否已安装transformers
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                model_name = "Qwen/Qwen3-Embedding-0.6B"
                
                # 加载模型和tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                
                # 设置为评估模式
                self.embedding_model.eval()
                
                logger.info("Qwen3-Embedding模型加载成功")
                return True
                
            except ImportError:
                logger.warning("transformers未安装，尝试使用sentence-transformers...")
                
                # 备选方案：使用sentence-transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    # 使用一个轻量级的替代模型进行测试
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("使用SentenceTransformer替代模型")
                    return True
                    
                except ImportError:
                    logger.error("请安装transformers或sentence-transformers库")
                    return False
                    
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def load_course_data(self) -> List[Dict[str, Any]]:
        """
        加载课程数据
        
        Returns:
            List[Dict]: 课程数据列表
        """
        logger.info("正在加载课程数据...")
        
        course_files = list(self.course_data_dir.glob("*.json"))
        
        if self.test_mode:
            course_files = course_files[:self.test_sample_size]
            logger.info(f"测试模式：加载前 {len(course_files)} 个文件")
        
        courses = []
        
        for file_path in tqdm(course_files, desc="加载课程文件"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    course_data = json.load(f)
                
                # 提取关键信息
                course_info = {
                    'file_name': file_path.stem,
                    'course_name': course_data.get('course_name', ''),
                    'course_description': course_data.get('course_description', ''),
                    'topics': course_data.get('topics', []),
                    'syllabus_content': course_data.get('syllabus_content', ''),
                    'files': course_data.get('files', [])
                }
                
                # 创建用于embedding的文本
                course_info['embedding_text'] = self._create_embedding_text(course_info)
                
                courses.append(course_info)
                
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")
                continue
        
        logger.info(f"成功加载 {len(courses)} 个课程")
        self.courses = courses
        return courses
    
    def _create_embedding_text(self, course_info: Dict[str, Any]) -> str:
        """
        创建用于embedding的综合文本
        
        Args:
            course_info: 课程信息
            
        Returns:
            str: 综合文本
        """
        text_parts = []
        
        # 课程名称（权重最高）
        if course_info['course_name']:
            text_parts.append(f"Course: {course_info['course_name']}")
        
        # 主题标签
        if course_info['topics']:
            unique_topics = list(set(course_info['topics']))  # 去重
            text_parts.append(f"Topics: {', '.join(unique_topics)}")
        
        # 课程描述
        if course_info['course_description']:
            # 截取前500字符，避免文本过长
            description = course_info['course_description'][:500]
            text_parts.append(f"Description: {description}")
        
        # 教学大纲（关键部分）
        if course_info['syllabus_content']:
            # 提取教学大纲的关键部分，避免过长
            syllabus = course_info['syllabus_content'][:800]
            text_parts.append(f"Syllabus: {syllabus}")
        
        return " | ".join(text_parts)
    
    def generate_embeddings(self) -> np.ndarray:
        """
        生成课程嵌入向量
        
        Returns:
            np.ndarray: 嵌入矩阵
        """
        if not self.courses:
            raise ValueError("请先加载课程数据")
        
        if not self.embedding_model:
            if not self._load_embedding_model():
                raise RuntimeError("模型加载失败")
        
        logger.info("正在生成embedding...")
        
        texts = [course['embedding_text'] for course in self.courses]
        
        try:
            # 根据模型类型选择不同的embedding方法
            if hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer
                embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            else:
                # Transformers模型
                embeddings = self._encode_with_transformers(texts)
            
            self.embeddings = embeddings
            
            # 保存embeddings
            embedding_file = self.output_dir / "embeddings" / f"course_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(embedding_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'courses': self.courses,
                    'metadata': {
                        'model': 'Qwen3-Embedding-0.6B' if not hasattr(self.embedding_model, 'encode') else 'SentenceTransformer',
                        'num_courses': len(self.courses),
                        'embedding_dim': embeddings.shape[1],
                        'created_at': datetime.now().isoformat()
                    }
                }, f)
            
            logger.info(f"Embedding生成完成: {embeddings.shape}")
            logger.info(f"已保存到: {embedding_file}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding生成失败: {e}")
            raise e
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """使用transformers模型进行编码"""
        import torch
        
        embeddings = []
        batch_size = 8  # 避免内存不足
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="生成embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # 分词
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
                
                # 获取embedding
                outputs = self.embedding_model(**inputs)
                
                # 使用mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def build_knowledge_graph(self, similarity_threshold: float = 0.7) -> nx.Graph:
        """
        构建知识图谱
        
        Args:
            similarity_threshold: 相似度阈值
            
        Returns:
            nx.Graph: 知识图谱
        """
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")
        
        logger.info("正在构建知识图谱...")
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for i, course in enumerate(self.courses):
            G.add_node(i, 
                      name=course['course_name'],
                      topics=course['topics'],
                      description=course['course_description'][:100] + "..." if len(course['course_description']) > 100 else course['course_description'])
        
        # 添加边（基于相似度）
        num_edges = 0
        for i in range(len(self.courses)):
            for j in range(i+1, len(self.courses)):
                similarity = similarity_matrix[i][j]
                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
                    num_edges += 1
        
        self.knowledge_graph = G
        
        logger.info(f"知识图谱构建完成: {len(G.nodes)} 个节点, {num_edges} 条边")
        
        # 保存图
        graph_file = self.output_dir / "graphs" / f"course_knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gml"
        
        # 为GML格式转换权重为字符串
        G_copy = G.copy()
        for u, v, d in G_copy.edges(data=True):
            if 'weight' in d:
                d['weight'] = str(d['weight'])
        
        nx.write_gml(G_copy, graph_file)
        logger.info(f"图谱已保存到: {graph_file}")
        
        # 同时保存为pickle格式（保留完整数据）
        pickle_file = self.output_dir / "graphs" / f"course_knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"图谱数据已保存到: {pickle_file}")
        
        return G
    
    def analyze_topics(self) -> Dict[str, Any]:
        """
        分析主题分布和聚类
        
        Returns:
            Dict: 分析结果
        """
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")
        
        logger.info("正在进行主题分析...")
        
        # 主题统计
        all_topics = []
        for course in self.courses:
            all_topics.extend(course['topics'])
        
        topic_counts = pd.Series(all_topics).value_counts()
        
        # K-means聚类
        n_clusters = min(10, len(self.courses) // 5)  # 动态确定聚类数
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # 分析每个聚类的主要主题
            cluster_topics = {}
            for i in range(n_clusters):
                cluster_courses = [self.courses[j] for j in range(len(self.courses)) if cluster_labels[j] == i]
                cluster_topic_list = []
                for course in cluster_courses:
                    cluster_topic_list.extend(course['topics'])
                
                if cluster_topic_list:
                    cluster_topics[f"Cluster_{i}"] = pd.Series(cluster_topic_list).value_counts().head(5).to_dict()
        else:
            cluster_labels = np.zeros(len(self.courses))
            cluster_topics = {}
        
        analysis_results = {
            'topic_distribution': topic_counts.head(20).to_dict(),
            'cluster_analysis': cluster_topics,
            'num_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist()
        }
        
        # 保存分析结果
        analysis_file = self.output_dir / "analysis" / f"topic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"主题分析完成，结果保存到: {analysis_file}")
        
        return analysis_results
    
    def visualize_knowledge_graph(self, max_nodes: int = 100):
        """
        可视化知识图谱
        
        Args:
            max_nodes: 最大显示节点数
        """
        if self.knowledge_graph is None:
            raise ValueError("请先构建知识图谱")
        
        logger.info("正在生成知识图谱可视化...")
        
        G = self.knowledge_graph
        
        # 如果节点太多，选择连接度最高的节点
        if len(G.nodes) > max_nodes:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([node for node, _ in top_nodes])
        
        plt.figure(figsize=(15, 10))
        
        # 使用spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        node_sizes = [300 + G.degree(node) * 50 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
        
        # 添加标签（只显示部分）
        if len(G.nodes) <= 50:
            labels = {node: G.nodes[node]['name'][:20] + "..." if len(G.nodes[node]['name']) > 20 else G.nodes[node]['name'] 
                     for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Course Knowledge Graph", fontsize=16)
        plt.axis('off')
        
        # 保存图片
        viz_file = self.output_dir / "graphs" / f"knowledge_graph_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图片保存到: {viz_file}")
    
    def visualize_topic_analysis(self, analysis_results: Dict[str, Any]):
        """
        可视化主题分析结果
        
        Args:
            analysis_results: 分析结果
        """
        logger.info("正在生成主题分析可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 主题分布条形图
        topic_dist = analysis_results['topic_distribution']
        topics = list(topic_dist.keys())[:15]  # 显示前15个
        counts = [topic_dist[topic] for topic in topics]
        
        axes[0,0].barh(range(len(topics)), counts)
        axes[0,0].set_yticks(range(len(topics)))
        axes[0,0].set_yticklabels(topics, fontsize=8)
        axes[0,0].set_title('Top Topics Distribution')
        axes[0,0].set_xlabel('Count')
        
        # 2. 聚类分析
        if self.embeddings is not None and len(analysis_results['cluster_labels']) > 1:
            cluster_labels = np.array(analysis_results['cluster_labels'])
            
            # 使用t-SNE降维可视化（如果数据量不大）
            if len(self.embeddings) <= 1000:
                try:
                    from sklearn.manifold import TSNE
                    # 调整perplexity参数，确保小于样本数
                    perplexity = min(30, len(self.embeddings) - 1, 5)
                    if perplexity >= 2:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                        embeddings_2d = tsne.fit_transform(self.embeddings)
                        
                        scatter = axes[0,1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                                  c=cluster_labels, cmap='tab10', alpha=0.7)
                        axes[0,1].set_title('Course Clusters (t-SNE)')
                        plt.colorbar(scatter, ax=axes[0,1])
                    else:
                        axes[0,1].text(0.5, 0.5, 'Too few samples for t-SNE visualization', 
                                      ha='center', va='center', transform=axes[0,1].transAxes)
                except ImportError:
                    axes[0,1].text(0.5, 0.5, 'sklearn.manifold.TSNE not available', 
                                  ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. 聚类主题词云（文本形式）
        cluster_analysis = analysis_results['cluster_analysis']
        if cluster_analysis:
            cluster_text = ""
            for cluster, topics in list(cluster_analysis.items())[:5]:  # 显示前5个聚类
                cluster_text += f"{cluster}:\n"
                for topic, count in list(topics.items())[:3]:  # 每个聚类显示前3个主题
                    cluster_text += f"  {topic}: {count}\n"
                cluster_text += "\n"
            
            axes[1,0].text(0.05, 0.95, cluster_text, fontsize=10, verticalalignment='top',
                          transform=axes[1,0].transAxes, fontfamily='monospace')
            axes[1,0].set_title('Cluster Topic Analysis')
            axes[1,0].axis('off')
        
        # 4. 统计信息
        stats_text = f"""
Dataset Statistics:
- Total Courses: {len(self.courses)}
- Unique Topics: {len(analysis_results['topic_distribution'])}
- Number of Clusters: {analysis_results['num_clusters']}
- Embedding Dimension: {self.embeddings.shape[1] if self.embeddings is not None else 'N/A'}
- Graph Nodes: {len(self.knowledge_graph.nodes) if self.knowledge_graph else 'N/A'}
- Graph Edges: {len(self.knowledge_graph.edges) if self.knowledge_graph else 'N/A'}
"""
        
        axes[1,1].text(0.05, 0.95, stats_text, fontsize=12, verticalalignment='top',
                      transform=axes[1,1].transAxes, fontfamily='monospace')
        axes[1,1].set_title('Dataset Statistics')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        viz_file = self.output_dir / "analysis" / f"topic_analysis_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"主题分析可视化保存到: {viz_file}")
    
    def generate_summary_report(self) -> str:
        """
        生成总结报告
        
        Returns:
            str: 报告内容
        """
        report = f"""
# Course Knowledge Graph Analysis Report

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Total Courses Processed**: {len(self.courses)}
- **Test Mode**: {'Yes' if self.test_mode else 'No'}
- **Sample Size**: {self.test_sample_size if self.test_mode else 'All'}

## Embedding Analysis
- **Model Used**: {'Qwen3-Embedding-0.6B' if not hasattr(self.embedding_model, 'encode') else 'SentenceTransformer (fallback)'}
- **Embedding Dimension**: {self.embeddings.shape[1] if self.embeddings is not None else 'N/A'}
- **Average Text Length**: {np.mean([len(course['embedding_text']) for course in self.courses]):.1f} characters

## Knowledge Graph Statistics
- **Nodes**: {len(self.knowledge_graph.nodes) if self.knowledge_graph else 'N/A'}
- **Edges**: {len(self.knowledge_graph.edges) if self.knowledge_graph else 'N/A'}
- **Average Degree**: {np.mean(list(dict(self.knowledge_graph.degree()).values())) if self.knowledge_graph else 'N/A'}
- **Graph Density**: {nx.density(self.knowledge_graph) if self.knowledge_graph else 'N/A'}

## Top Course Topics
"""
        
        # 添加主题统计
        if self.courses:
            all_topics = []
            for course in self.courses:
                all_topics.extend(course['topics'])
            
            topic_counts = pd.Series(all_topics).value_counts().head(10)
            for topic, count in topic_counts.items():
                report += f"- **{topic}**: {count} courses\n"
        
        report += f"""

## Output Files Generated
- Embeddings: `{self.output_dir}/embeddings/`
- Knowledge Graph: `{self.output_dir}/graphs/`
- Analysis Results: `{self.output_dir}/analysis/`

## Next Steps
1. Review the knowledge graph visualization
2. Analyze topic clusters for curriculum alignment
3. Use similarity scores for course recommendation
4. Export data for further analysis or web interface

---
*Generated by DCAS Course Knowledge Graph Builder*
"""
        
        # 保存报告
        report_file = self.output_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"总结报告保存到: {report_file}")
        
        return report

def main():
    """主函数"""
    # 配置
    course_data_dir = "datasets/Course Details/General"
    output_dir = "knowledge_graph_output"
    
    # 检测是否为本地测试环境
    test_mode = True  # 本地测试，采样少量数据
    test_sample_size = 20  # 测试样本数量
    
    print("=" * 60)
    print("🎓 DCAS Course Knowledge Graph Builder")
    print("=" * 60)
    
    try:
        # 初始化构建器
        builder = CourseKnowledgeGraphBuilder(
            course_data_dir=course_data_dir,
            output_dir=output_dir,
            test_mode=test_mode,
            test_sample_size=test_sample_size
        )
        
        # 1. 加载课程数据
        print("\n📚 Step 1: Loading course data...")
        courses = builder.load_course_data()
        
        if not courses:
            logger.error("没有加载到课程数据，请检查数据目录")
            return
        
        print(f"✅ Loaded {len(courses)} courses")
        
        # 2. 生成embeddings
        print("\n🔢 Step 2: Generating embeddings...")
        embeddings = builder.generate_embeddings()
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        
        # 3. 构建知识图谱
        print("\n🕸️  Step 3: Building knowledge graph...")
        knowledge_graph = builder.build_knowledge_graph(similarity_threshold=0.6)
        print(f"✅ Built graph with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
        
        # 4. 主题分析
        print("\n📊 Step 4: Analyzing topics...")
        analysis_results = builder.analyze_topics()
        print(f"✅ Identified {len(analysis_results['topic_distribution'])} unique topics")
        
        # 5. 生成可视化
        print("\n🎨 Step 5: Generating visualizations...")
        builder.visualize_knowledge_graph()
        builder.visualize_topic_analysis(analysis_results)
        print("✅ Visualizations generated")
        
        # 6. 生成报告
        print("\n📋 Step 6: Generating summary report...")
        report = builder.generate_summary_report()
        print("✅ Summary report generated")
        
        print(f"\n🎉 All done! Results saved to: {output_dir}")
        
        # 显示主要统计信息
        print("\n📈 Quick Stats:")
        print(f"  - Courses processed: {len(courses)}")
        print(f"  - Embedding dimensions: {embeddings.shape[1]}")
        print(f"  - Graph connections: {len(knowledge_graph.edges)}")
        print(f"  - Top topics: {list(analysis_results['topic_distribution'].keys())[:3]}")
        
        if test_mode:
            print(f"\n⚠️  Note: Running in test mode with {test_sample_size} samples")
            print("   To process all data, set test_mode=False and run on server")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
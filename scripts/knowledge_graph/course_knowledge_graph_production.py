<<<<<<< HEAD
#!/usr/bin/env python3
"""
DCAS - 课程知识图谱构建器 (生产版本)

服务器部署版本：处理全量数据，优化内存使用，支持断点续传
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
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionCourseKnowledgeGraphBuilder:
    """生产环境课程知识图谱构建器"""
    
    def __init__(self, course_data_dir: str, output_dir: str = "knowledge_graph_output", 
                 batch_size: int = 100, max_memory_gb: float = 8.0, resume: bool = True):
        """
        初始化生产环境构建器
        
        Args:
            course_data_dir: 课程数据目录
            output_dir: 输出目录
            batch_size: 批处理大小
            max_memory_gb: 最大内存使用限制(GB)
            resume: 是否支持断点续传
        """
        self.course_data_dir = Path(course_data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.resume = resume
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # 检查点文件
        self.checkpoint_file = self.output_dir / "checkpoints" / "progress.json"
        self.course_cache_file = self.output_dir / "checkpoints" / "courses_cache.pkl"
        self.embedding_cache_file = self.output_dir / "checkpoints" / "embeddings_cache.pkl"
        
        # 数据存储
        self.courses = []
        self.embeddings = None
        self.knowledge_graph = None
        self.embedding_model = None
        
        logger.info(f"生产环境初始化完成 - 批大小: {batch_size}, 内存限��: {max_memory_gb}GB")
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        
        if memory_gb > self.max_memory_gb:
            logger.warning(f"内存使用过高: {memory_gb:.2f}GB > {self.max_memory_gb}GB，执行垃圾回收")
            gc.collect()
            
        return memory_gb
    
    def _save_checkpoint(self, stage: str, data: Dict[str, Any]):
        """保存检查点"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'memory_usage_gb': self._check_memory_usage(),
            **data
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查点已保存: {stage}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if self.resume and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                logger.info(f"加载检查点: {checkpoint['stage']} ({checkpoint['timestamp']})")
                return checkpoint
            except Exception as e:
                logger.warning(f"加载检查点失败: {e}")
        return None
    
    def _load_embedding_model(self):
        """加载embedding模型"""
        if self.embedding_model is not None:
            return True
            
        try:
            logger.info("正在加载Qwen3-Embedding-0.6B模型...")
            
            # 设置环境变量避免警告
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                model_name = "Qwen/Qwen3-Embedding-0.6B"
                
                # 检查GPU可用性
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"使用设备: {device}")
                
                # 加载模型和tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.embedding_model = AutoModel.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                ).to(device)
                
                # 设置为评估模式
                self.embedding_model.eval()
                
                logger.info(f"Qwen3-Embedding模型加载成功 (设备: {device})")
                return True
                
            except ImportError:
                logger.warning("transformers未安装，使用sentence-transformers备选方案...")
                
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("使用SentenceTransformer备选模型")
                return True
                    
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def load_course_data(self) -> List[Dict[str, Any]]:
        """批量加载课程数据"""
        checkpoint = self._load_checkpoint()
        
        # 检查是否可以从缓存加载
        if (checkpoint and checkpoint['stage'] in ['data_loaded', 'embeddings_generated', 'completed'] 
            and self.course_cache_file.exists()):
            try:
                with open(self.course_cache_file, 'rb') as f:
                    self.courses = pickle.load(f)
                logger.info(f"从缓存加载了 {len(self.courses)} 个课程")
                return self.courses
            except Exception as e:
                logger.warning(f"缓存加载失败，重新处理: {e}")
        
        logger.info("正在加载课程数据...")
        
        course_files = list(self.course_data_dir.glob("*.json"))
        logger.info(f"发现 {len(course_files)} 个课程文件")
        
        courses = []
        processed_count = 0
        
        # 批量处理文件
        for i in tqdm(range(0, len(course_files), self.batch_size), desc="批量加载课程"):
            batch_files = course_files[i:i+self.batch_size]
            
            for file_path in batch_files:
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
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {e}")
                    continue
            
            # 检查内存使用
            self._check_memory_usage()
            
            # 定期保存进度
            if processed_count % (self.batch_size * 10) == 0:
                logger.info(f"已处理 {processed_count} 个文件...")
        
        logger.info(f"成功加载 {len(courses)} 个课程")
        self.courses = courses
        
        # 保存缓存
        with open(self.course_cache_file, 'wb') as f:
            pickle.dump(courses, f)
        
        # 保存检查点
        self._save_checkpoint('data_loaded', {'num_courses': len(courses)})
        
        return courses
    
    def _create_embedding_text(self, course_info: Dict[str, Any]) -> str:
        """创建用于embedding的综合文本"""
        text_parts = []
        
        # 课程名称（权重最高）
        if course_info['course_name']:
            text_parts.append(f"Course: {course_info['course_name']}")
        
        # 主题标签
        if course_info['topics']:
            unique_topics = list(set(course_info['topics']))
            text_parts.append(f"Topics: {', '.join(unique_topics)}")
        
        # 课程描述（截取前300字符）
        if course_info['course_description']:
            description = course_info['course_description'][:300]
            text_parts.append(f"Description: {description}")
        
        # 教学大纲（截取前500字符）
        if course_info['syllabus_content']:
            syllabus = course_info['syllabus_content'][:500]
            text_parts.append(f"Syllabus: {syllabus}")
        
        return " | ".join(text_parts)
    
    def generate_embeddings(self) -> np.ndarray:
        """批量生成课程嵌入向量"""
        checkpoint = self._load_checkpoint()
        
        # 检查是否可以从缓存加载
        if (checkpoint and checkpoint['stage'] in ['embeddings_generated', 'completed'] 
            and self.embedding_cache_file.exists()):
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    self.embeddings = embeddings_data['embeddings']
                logger.info(f"从缓存加载embeddings: {self.embeddings.shape}")
                return self.embeddings
            except Exception as e:
                logger.warning(f"embedding缓存加载失败，重新生成: {e}")
        
        if not self.courses:
            raise ValueError("请先加载课程数据")
        
        if not self._load_embedding_model():
            raise RuntimeError("模型加载失败")
        
        logger.info("正在批量生成embedding...")
        
        texts = [course['embedding_text'] for course in self.courses]
        
        try:
            # 批量处理embedding
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), self.batch_size), desc="生成embeddings"):
                batch_texts = texts[i:i+self.batch_size]
                
                if hasattr(self.embedding_model, 'encode'):
                    # SentenceTransformer
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts, 
                        show_progress_bar=False,
                        batch_size=min(32, len(batch_texts))
                    )
                else:
                    # Transformers模型
                    batch_embeddings = self._encode_with_transformers(batch_texts)
                
                all_embeddings.append(batch_embeddings)
                
                # 检查内存使用
                self._check_memory_usage()
            
            # 合并所有embeddings
            embeddings = np.vstack(all_embeddings)
            self.embeddings = embeddings
            
            # 保存embeddings缓存
            embeddings_data = {
                'embeddings': embeddings,
                'metadata': {
                    'model': 'Qwen3-Embedding-0.6B' if not hasattr(self.embedding_model, 'encode') else 'SentenceTransformer',
                    'num_courses': len(self.courses),
                    'embedding_dim': embeddings.shape[1],
                    'created_at': datetime.now().isoformat()
                }
            }
            
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            # 保存到最终输出目录
            embedding_file = self.output_dir / "embeddings" / f"course_embeddings_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(embedding_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'courses': self.courses,
                    **embeddings_data['metadata']
                }, f)
            
            logger.info(f"Embedding生成完成: {embeddings.shape}")
            logger.info(f"已保存到: {embedding_file}")
            
            # 保存检查点
            self._save_checkpoint('embeddings_generated', {
                'embedding_shape': embeddings.shape,
                'embedding_file': str(embedding_file)
            })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding生成失败: {e}")
            raise e
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """使用transformers模型进行编码"""
        import torch
        
        embeddings = []
        mini_batch_size = 8  # 减小批大小避免内存溢出
        
        with torch.no_grad():
            for i in range(0, len(texts), mini_batch_size):
                batch_texts = texts[i:i+mini_batch_size]
                
                # 分词
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
                
                # 移动到正确的设备
                device = next(self.embedding_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 获取embedding
                outputs = self.embedding_model(**inputs)
                
                # 使用mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def build_knowledge_graph(self, similarity_threshold: float = 0.92,  # 进一步提高阈值 
                            max_edges: int = 50000) -> nx.Graph:  # 减少最大边数
        """构建知识图谱（内存优化版本）"""
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")
        
        logger.info("正在构建知识图谱...")
        
        # 对于大型数据集，分块计算相似度
        num_courses = len(self.courses)
        logger.info(f"计算 {num_courses} 个课程的相似度矩阵...")
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for i, course in enumerate(self.courses):
            G.add_node(i, 
                      name=course['course_name'],
                      topics=course['topics'][:5],  # 限制主题数量
                      description=course['course_description'][:100] + "..." if len(course['course_description']) > 100 else course['course_description'])
        
        # 分块计算相似度并添加边
        chunk_size = min(1000, num_courses)  # 动态调整块大小
        edge_count = 0
        max_edges_reached = False
        
        for i in tqdm(range(0, num_courses, chunk_size), desc="计算相似度"):
            if max_edges_reached:
                break
                
            end_i = min(i + chunk_size, num_courses)
            chunk_embeddings_i = self.embeddings[i:end_i]
            
            for j in range(i, num_courses, chunk_size):
                if max_edges_reached:
                    break
                    
                end_j = min(j + chunk_size, num_courses)
                chunk_embeddings_j = self.embeddings[j:end_j]
                
                # 计算这两个块之间的相似度
                similarity_chunk = cosine_similarity(chunk_embeddings_i, chunk_embeddings_j)
                
                # 添加边
                for ii in range(similarity_chunk.shape[0]):
                    for jj in range(similarity_chunk.shape[1]):
                        global_i = i + ii
                        global_j = j + jj
                        
                        if global_i >= global_j:  # 避免重复和自连接
                            continue
                            
                        similarity = similarity_chunk[ii, jj]
                        if similarity > similarity_threshold:
                            G.add_edge(global_i, global_j, weight=float(similarity))
                            edge_count += 1
                            
                            # 限制边数以控制内存使用
                            if edge_count >= max_edges:
                                if not max_edges_reached:  # 只在第一次达到限制时记录警告
                                    logger.warning(f"达到最大边数限制 {max_edges}，停止添加边")
                                max_edges_reached = True
                                break
                    
                    if max_edges_reached:
                        break
            
            # 检查内存使用
            if not max_edges_reached:
                self._check_memory_usage()
        
        self.knowledge_graph = G
        
        logger.info(f"知识图谱构建完成: {len(G.nodes)} 个节点, {edge_count} 条边")
        
        # 保存图
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存为pickle格式（保留完整数据）
        pickle_file = self.output_dir / "graphs" / f"course_knowledge_graph_production_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"图谱数据已保存到: {pickle_file}")
        
        # 保存检查点
        self._save_checkpoint('graph_built', {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'graph_file': str(pickle_file)
        })
        
        return G
    
    def analyze_topics(self) -> Dict[str, Any]:
        """分析主题分布和聚类（内存优化版本）"""
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")
        
        logger.info("正在进行主题分析...")
        
        # 主题统计
        all_topics = []
        for course in self.courses:
            all_topics.extend(course['topics'])
        
        topic_counts = pd.Series(all_topics).value_counts()
        
        # 使用MiniBatchKMeans进行聚类（内存友好）
        n_clusters = min(20, len(self.courses) // 10)
        if n_clusters > 1:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
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
            'topic_distribution': topic_counts.head(50).to_dict(),  # 增加到50个主题
            'cluster_analysis': cluster_topics,
            'num_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist()
        }
        
        # 保存分析结果
        analysis_file = self.output_dir / "analysis" / f"topic_analysis_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"主题分析完成，结果保存��: {analysis_file}")
        
        return analysis_results
    
    def generate_summary_report(self) -> str:
        """生成生产环境总结报告"""
        memory_usage = self._check_memory_usage()
        
        report = f"""
# DCAS Course Knowledge Graph - Production Report

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Production Environment Stats
- **Processing Mode**: Full Dataset (Production)
- **Total Courses Processed**: {len(self.courses)}
- **Memory Usage**: {memory_usage:.2f} GB
- **Batch Size**: {self.batch_size}

## Embedding Analysis
- **Model Used**: {'Qwen3-Embedding-0.6B' if not hasattr(self.embedding_model, 'encode') else 'SentenceTransformer (fallback)'}
- **Embedding Dimension**: {self.embeddings.shape[1] if self.embeddings is not None else 'N/A'}
- **Processing Batches**: {(len(self.courses) + self.batch_size - 1) // self.batch_size}

## Knowledge Graph Statistics
- **Nodes**: {len(self.knowledge_graph.nodes) if self.knowledge_graph else 'N/A'}
- **Edges**: {len(self.knowledge_graph.edges) if self.knowledge_graph else 'N/A'}
- **Graph Density**: {(nx.density(self.knowledge_graph) if self.knowledge_graph else 0):.6f}

## Performance Metrics
- **Average Text Length**: {np.mean([len(course['embedding_text']) for course in self.courses]):.1f} characters
- **Courses per Batch**: {self.batch_size}
- **Memory Efficiency**: Optimized for large datasets

## Top Course Topics (Production Scale)
"""
        
        # 添加主题统计
        if self.courses:
            all_topics = []
            for course in self.courses:
                all_topics.extend(course['topics'])
            
            topic_counts = pd.Series(all_topics).value_counts().head(20)
            for topic, count in topic_counts.items():
                report += f"- **{topic}**: {count} courses\n"
        
        report += f"""

## Output Files (Production)
- **Embeddings**: `{self.output_dir}/embeddings/`
- **Knowledge Graph**: `{self.output_dir}/graphs/`
- **Analysis Results**: `{self.output_dir}/analysis/`
- **Checkpoints**: `{self.output_dir}/checkpoints/`

## Deployment Recommendations
1. **Server Requirements**: Minimum 16GB RAM for full dataset
2. **GPU Acceleration**: CUDA-compatible GPU recommended
3. **Storage**: SSD recommended for faster I/O
4. **Monitoring**: Track memory usage during processing

## API Integration Ready
- All outputs saved in pickle format for easy loading
- Embedding vectors can be used for real-time similarity search
- Knowledge graph ready for web visualization

---
*Generated by DCAS Production Course Knowledge Graph Builder*
*Server optimized for datasets of {len(self.courses)} courses*
"""
        
        # 保存报告
        report_file = self.output_dir / f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"生产环境报告保存到: {report_file}")
        
        # 保存最终检查点
        self._save_checkpoint('completed', {
            'report_file': str(report_file),
            'final_memory_usage': memory_usage
        })
        
        return report

def main():
    """生产环境主函数"""
    parser = argparse.ArgumentParser(description='DCAS Course Knowledge Graph Builder - Production')
    parser.add_argument('--data-dir', default='datasets/Course Details/General', 
                       help='课程数据目录')
    parser.add_argument('--output-dir', default='knowledge_graph_output_production', 
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='批处理大小')
    parser.add_argument('--max-memory', type=float, default=12.0, 
                       help='最大内存使用(GB)')
    parser.add_argument('--similarity-threshold', type=float, default=0.85, 
                       help='相似度阈值 (推荐: 0.85-0.9)')
    parser.add_argument('--no-resume', action='store_true', 
                       help='不使用断点续传')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 DCAS Course Knowledge Graph Builder - PRODUCTION MODE")
    print("=" * 80)
    
    try:
        # 初始化构建器
        builder = ProductionCourseKnowledgeGraphBuilder(
            course_data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_memory_gb=args.max_memory,
            resume=not args.no_resume
        )
        
        # 1. 加载课程数据
        print(f"\n📚 Step 1: Loading course data (batch size: {args.batch_size})...")
        courses = builder.load_course_data()
        print(f"✅ Loaded {len(courses)} courses")
        
        # 2. 生成embeddings
        print(f"\n🔢 Step 2: Generating embeddings...")
        embeddings = builder.generate_embeddings()
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # 3. 构建知识图谱
        print(f"\n🕸️  Step 3: Building knowledge graph (threshold: {args.similarity_threshold})...")
        knowledge_graph = builder.build_knowledge_graph(similarity_threshold=args.similarity_threshold)
        print(f"✅ Built graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        
        # 4. 主题分析
        print(f"\n📊 Step 4: Analyzing topics...")
        analysis_results = builder.analyze_topics()
        print(f"✅ Analyzed {len(analysis_results['topic_distribution'])} topics")
        
        # 5. 生成报告
        print(f"\n📋 Step 5: Generating production report...")
        report = builder.generate_summary_report()
        print("✅ Production report generated")
        
        print(f"\n🎉 Production pipeline completed!")
        print(f"📁 Results: {args.output_dir}")
        print(f"💾 Memory used: {builder._check_memory_usage():.2f} GB")
        
        # 显示关键统计
        print(f"\n📈 Production Stats:")
        print(f"  - Total courses: {len(courses)}")
        print(f"  - Embedding dims: {embeddings.shape[1]}")
        print(f"  - Graph edges: {len(knowledge_graph.edges)}")
        print(f"  - Processing batches: {(len(courses) + args.batch_size - 1) // args.batch_size}")
        
    except Exception as e:
        logger.error(f"生产环境执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python3
"""
DCAS - 课程知识图谱构建器 (生产版本)

服务器部署版本：处理全量数据，优化内存使用，支持断点续传
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
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionCourseKnowledgeGraphBuilder:
    """生产环境课程知识图谱构建器"""
    
    def __init__(self, course_data_dir: str, output_dir: str = "knowledge_graph_output", 
                 batch_size: int = 100, max_memory_gb: float = 8.0, resume: bool = True):
        """
        初始化生产环境构建器
        
        Args:
            course_data_dir: 课程数据目录
            output_dir: 输出目录
            batch_size: 批处理大小
            max_memory_gb: 最大内存使用限制(GB)
            resume: 是否支持断点续传
        """
        self.course_data_dir = Path(course_data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.resume = resume
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # 检查点文件
        self.checkpoint_file = self.output_dir / "checkpoints" / "progress.json"
        self.course_cache_file = self.output_dir / "checkpoints" / "courses_cache.pkl"
        self.embedding_cache_file = self.output_dir / "checkpoints" / "embeddings_cache.pkl"
        
        # 数据存储
        self.courses = []
        self.embeddings = None
        self.knowledge_graph = None
        self.embedding_model = None
        
        logger.info(f"生产环境初始化完成 - 批大小: {batch_size}, 内存限制: {max_memory_gb}GB")
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        
        if memory_gb > self.max_memory_gb:
            logger.warning(f"内存使用过高: {memory_gb:.2f}GB > {self.max_memory_gb}GB，执行垃圾回收")
            gc.collect()
            
        return memory_gb
    
    def _save_checkpoint(self, stage: str, data: Dict[str, Any]):
        """保存检查点"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'memory_usage_gb': self._check_memory_usage(),
            **data
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查点已保存: {stage}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if self.resume and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                logger.info(f"加载检查点: {checkpoint['stage']} ({checkpoint['timestamp']})")
                return checkpoint
            except Exception as e:
                logger.warning(f"加载检查点失败: {e}")
        return None
    
    def _load_embedding_model(self):
        """加载embedding模型"""
        if self.embedding_model is not None:
            return True
            
        try:
            logger.info("正在加载Qwen3-Embedding-0.6B模型...")
            
            # 设置环境变量避免警告
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                model_name = "Qwen/Qwen3-Embedding-0.6B"
                
                # 检查GPU可用性
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"使用设备: {device}")
                
                # 加载模型和tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.embedding_model = AutoModel.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                ).to(device)
                
                # 设置为评估模式
                self.embedding_model.eval()
                
                logger.info(f"Qwen3-Embedding模型加载成功 (设备: {device})")
                return True
                
            except ImportError:
                logger.warning("transformers未安装，使用sentence-transformers备选方案...")
                
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("使用SentenceTransformer备选模型")
                return True
                    
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def load_course_data(self) -> List[Dict[str, Any]]:
        """批量加载课程数据"""
        checkpoint = self._load_checkpoint()
        
        # 检查是否可以从缓存加载
        if (checkpoint and checkpoint['stage'] in ['data_loaded', 'embeddings_generated', 'completed'] 
            and self.course_cache_file.exists()):
            try:
                with open(self.course_cache_file, 'rb') as f:
                    self.courses = pickle.load(f)
                logger.info(f"从缓存加载了 {len(self.courses)} 个课程")
                return self.courses
            except Exception as e:
                logger.warning(f"缓存加载失败，重新处理: {e}")
        
        logger.info("正在加载课程数据...")
        
        course_files = list(self.course_data_dir.glob("*.json"))
        logger.info(f"发现 {len(course_files)} 个课程文件")
        
        courses = []
        processed_count = 0
        
        # 批量处理文件
        for i in tqdm(range(0, len(course_files), self.batch_size), desc="批量加载课程"):
            batch_files = course_files[i:i+self.batch_size]
            
            for file_path in batch_files:
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
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {e}")
                    continue
            
            # 检查内存使用
            self._check_memory_usage()
            
            # 定期保存进度
            if processed_count % (self.batch_size * 10) == 0:
                logger.info(f"已处理 {processed_count} 个文件...")
        
        logger.info(f"成功加载 {len(courses)} 个课程")
        self.courses = courses
        
        # 保存缓存
        with open(self.course_cache_file, 'wb') as f:
            pickle.dump(courses, f)
        
        # 保存检查点
        self._save_checkpoint('data_loaded', {'num_courses': len(courses)})
        
        return courses
    
    def _create_embedding_text(self, course_info: Dict[str, Any]) -> str:
        """创建用于embedding的综合文本"""
        text_parts = []
        
        # 课程名称（权重最高）
        if course_info['course_name']:
            text_parts.append(f"Course: {course_info['course_name']}")
        
        # 主题标签
        if course_info['topics']:
            unique_topics = list(set(course_info['topics']))
            text_parts.append(f"Topics: {', '.join(unique_topics)}")
        
        # 课程描述（截取前300字符）
        if course_info['course_description']:
            description = course_info['course_description'][:300]
            text_parts.append(f"Description: {description}")
        
        # 教学大纲（截取前500字符）
        if course_info['syllabus_content']:
            syllabus = course_info['syllabus_content'][:500]
            text_parts.append(f"Syllabus: {syllabus}")
        
        return " | ".join(text_parts)
    
    def generate_embeddings(self) -> np.ndarray:
        """批量生成课程嵌入向量"""
        checkpoint = self._load_checkpoint()
        
        # 检查是否可以从缓存加载
        if (checkpoint and checkpoint['stage'] in ['embeddings_generated', 'completed'] 
            and self.embedding_cache_file.exists()):
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    self.embeddings = embeddings_data['embeddings']
                logger.info(f"从缓存加载embeddings: {self.embeddings.shape}")
                return self.embeddings
            except Exception as e:
                logger.warning(f"embedding缓存加载失败，重新生成: {e}")
        
        if not self.courses:
            raise ValueError("请先加载课程数据")
        
        if not self._load_embedding_model():
            raise RuntimeError("模型加载失败")
        
        logger.info("正在批量生成embedding...")
        
        texts = [course['embedding_text'] for course in self.courses]
        
        try:
            # 批量处理embedding
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), self.batch_size), desc="生成embeddings"):
                batch_texts = texts[i:i+self.batch_size]
                
                if hasattr(self.embedding_model, 'encode'):
                    # SentenceTransformer
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts, 
                        show_progress_bar=False,
                        batch_size=min(32, len(batch_texts))
                    )
                else:
                    # Transformers模型
                    batch_embeddings = self._encode_with_transformers(batch_texts)
                
                all_embeddings.append(batch_embeddings)
                
                # 检查内存使用
                self._check_memory_usage()
            
            # 合并所有embeddings
            embeddings = np.vstack(all_embeddings)
            self.embeddings = embeddings
            
            # 保存embeddings缓存
            embeddings_data = {
                'embeddings': embeddings,
                'metadata': {
                    'model': 'Qwen3-Embedding-0.6B' if not hasattr(self.embedding_model, 'encode') else 'SentenceTransformer',
                    'num_courses': len(self.courses),
                    'embedding_dim': embeddings.shape[1],
                    'created_at': datetime.now().isoformat()
                }
            }
            
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            # 保存到最终输出目录
            embedding_file = self.output_dir / "embeddings" / f"course_embeddings_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(embedding_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'courses': self.courses,
                    **embeddings_data['metadata']
                }, f)
            
            logger.info(f"Embedding生成完成: {embeddings.shape}")
            logger.info(f"已保存到: {embedding_file}")
            
            # 保存检查点
            self._save_checkpoint('embeddings_generated', {
                'embedding_shape': embeddings.shape,
                'embedding_file': str(embedding_file)
            })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding生成失败: {e}")
            raise e
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """使用transformers模型进行编码"""
        import torch
        
        embeddings = []
        mini_batch_size = 8  # 减小批大小避免内存溢出
        
        with torch.no_grad():
            for i in range(0, len(texts), mini_batch_size):
                batch_texts = texts[i:i+mini_batch_size]
                
                # 分词
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
                
                # 移动到正确的设备
                device = next(self.embedding_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 获取embedding
                outputs = self.embedding_model(**inputs)
                
                # 使用mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def build_knowledge_graph(self, similarity_threshold: float = 0.92,  # 进一步提高阈值 
                            max_edges: int = 50000) -> nx.Graph:  # 减少最大边数
        """构建知识图谱（内存优化版本）"""
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")
        
        logger.info("正在构建知识图谱...")
        
        # 对于大型数据集，分块计算相似度
        num_courses = len(self.courses)
        logger.info(f"计算 {num_courses} 个课程的相似度矩阵...")
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for i, course in enumerate(self.courses):
            G.add_node(i, 
                      name=course['course_name'],
                      topics=course['topics'][:5],  # 限制主题数量
                      description=course['course_description'][:100] + "..." if len(course['course_description']) > 100 else course['course_description'])
        
        # 分块计算相似度并添加边
        chunk_size = min(1000, num_courses)  # 动态调整块大小
        edge_count = 0
        max_edges_reached = False
        
        for i in tqdm(range(0, num_courses, chunk_size), desc="计算相似度"):
            if max_edges_reached:
                break
                
            end_i = min(i + chunk_size, num_courses)
            chunk_embeddings_i = self.embeddings[i:end_i]
            
            for j in range(i, num_courses, chunk_size):
                if max_edges_reached:
                    break
                    
                end_j = min(j + chunk_size, num_courses)
                chunk_embeddings_j = self.embeddings[j:end_j]
                
                # 计算这两个块之间的相似度
                similarity_chunk = cosine_similarity(chunk_embeddings_i, chunk_embeddings_j)
                
                # 添加边
                for ii in range(similarity_chunk.shape[0]):
                    for jj in range(similarity_chunk.shape[1]):
                        global_i = i + ii
                        global_j = j + jj
                        
                        if global_i >= global_j:  # 避免重复和自连接
                            continue
                            
                        similarity = similarity_chunk[ii, jj]
                        if similarity > similarity_threshold:
                            G.add_edge(global_i, global_j, weight=float(similarity))
                            edge_count += 1
                            
                            # 限制边数以控制内存使用
                            if edge_count >= max_edges:
                                if not max_edges_reached:  # 只在第一次达到限制时记录警告
                                    logger.warning(f"达到最大边数限制 {max_edges}，停止添加边")
                                max_edges_reached = True
                                break
                    
                    if max_edges_reached:
                        break
            
            # 检查内存使用
            if not max_edges_reached:
                self._check_memory_usage()
        
        self.knowledge_graph = G
        
        logger.info(f"知识图谱构建完成: {len(G.nodes)} 个节点, {edge_count} 条边")
        
        # 保存图
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存为pickle格式（保留完整数据）
        pickle_file = self.output_dir / "graphs" / f"course_knowledge_graph_production_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"图谱数据已保存到: {pickle_file}")
        
        # 保存检查点
        self._save_checkpoint('graph_built', {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'graph_file': str(pickle_file)
        })
        
        return G
    
    def analyze_topics(self) -> Dict[str, Any]:
        """分析主题分布和聚类（内存优化版本）"""
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")
        
        logger.info("正在进行主题分析...")
        
        # 主题统计
        all_topics = []
        for course in self.courses:
            all_topics.extend(course['topics'])
        
        topic_counts = pd.Series(all_topics).value_counts()
        
        # 使用MiniBatchKMeans进行聚类（内存友好）
        n_clusters = min(20, len(self.courses) // 10)
        if n_clusters > 1:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
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
            'topic_distribution': topic_counts.head(50).to_dict(),  # 增加到50个主题
            'cluster_analysis': cluster_topics,
            'num_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist()
        }
        
        # 保存分析结果
        analysis_file = self.output_dir / "analysis" / f"topic_analysis_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"主题分析完成，结果保存到: {analysis_file}")
        
        return analysis_results
    
    def generate_summary_report(self) -> str:
        """生成生产环境总结报告"""
        memory_usage = self._check_memory_usage()
        
        report = f"""
# DCAS Course Knowledge Graph - Production Report

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Production Environment Stats
- **Processing Mode**: Full Dataset (Production)
- **Total Courses Processed**: {len(self.courses)}
- **Memory Usage**: {memory_usage:.2f} GB
- **Batch Size**: {self.batch_size}

## Embedding Analysis
- **Model Used**: {'Qwen3-Embedding-0.6B' if not hasattr(self.embedding_model, 'encode') else 'SentenceTransformer (fallback)'}
- **Embedding Dimension**: {self.embeddings.shape[1] if self.embeddings is not None else 'N/A'}
- **Processing Batches**: {(len(self.courses) + self.batch_size - 1) // self.batch_size}

## Knowledge Graph Statistics
- **Nodes**: {len(self.knowledge_graph.nodes) if self.knowledge_graph else 'N/A'}
- **Edges**: {len(self.knowledge_graph.edges) if self.knowledge_graph else 'N/A'}
- **Graph Density**: {(nx.density(self.knowledge_graph) if self.knowledge_graph else 0):.6f}

## Performance Metrics
- **Average Text Length**: {np.mean([len(course['embedding_text']) for course in self.courses]):.1f} characters
- **Courses per Batch**: {self.batch_size}
- **Memory Efficiency**: Optimized for large datasets

## Top Course Topics (Production Scale)
"""
        
        # 添加主题统计
        if self.courses:
            all_topics = []
            for course in self.courses:
                all_topics.extend(course['topics'])
            
            topic_counts = pd.Series(all_topics).value_counts().head(20)
            for topic, count in topic_counts.items():
                report += f"- **{topic}**: {count} courses\n"
        
        report += f"""

## Output Files (Production)
- **Embeddings**: `{self.output_dir}/embeddings/`
- **Knowledge Graph**: `{self.output_dir}/graphs/`
- **Analysis Results**: `{self.output_dir}/analysis/`
- **Checkpoints**: `{self.output_dir}/checkpoints/`

## Deployment Recommendations
1. **Server Requirements**: Minimum 16GB RAM for full dataset
2. **GPU Acceleration**: CUDA-compatible GPU recommended
3. **Storage**: SSD recommended for faster I/O
4. **Monitoring**: Track memory usage during processing

## API Integration Ready
- All outputs saved in pickle format for easy loading
- Embedding vectors can be used for real-time similarity search
- Knowledge graph ready for web visualization

---
*Generated by DCAS Production Course Knowledge Graph Builder*
*Server optimized for datasets of {len(self.courses)} courses*
"""
        
        # 保存报告
        report_file = self.output_dir / f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"生产环境报告保存到: {report_file}")
        
        # 保存最终检查点
        self._save_checkpoint('completed', {
            'report_file': str(report_file),
            'final_memory_usage': memory_usage
        })
        
        return report

def main():
    """生产环境主函数"""
    parser = argparse.ArgumentParser(description='DCAS Course Knowledge Graph Builder - Production')
    parser.add_argument('--data-dir', default='datasets/Course Details/General', 
                       help='课程数据目录')
    parser.add_argument('--output-dir', default='knowledge_graph_output_production', 
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='批处理大小')
    parser.add_argument('--max-memory', type=float, default=12.0, 
                       help='最大内存使用(GB)')
    parser.add_argument('--similarity-threshold', type=float, default=0.85, 
                       help='相似度阈值 (推荐: 0.85-0.9)')
    parser.add_argument('--no-resume', action='store_true', 
                       help='不使用断点续传')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 DCAS Course Knowledge Graph Builder - PRODUCTION MODE")
    print("=" * 80)
    
    try:
        # 初始化构建器
        builder = ProductionCourseKnowledgeGraphBuilder(
            course_data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_memory_gb=args.max_memory,
            resume=not args.no_resume
        )
        
        # 1. 加载课程数据
        print(f"\n📚 Step 1: Loading course data (batch size: {args.batch_size})...")
        courses = builder.load_course_data()
        print(f"✅ Loaded {len(courses)} courses")
        
        # 2. 生成embeddings
        print(f"\n🔢 Step 2: Generating embeddings...")
        embeddings = builder.generate_embeddings()
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # 3. 构建知识图谱
        print(f"\n🕸️  Step 3: Building knowledge graph (threshold: {args.similarity_threshold})...")
        knowledge_graph = builder.build_knowledge_graph(similarity_threshold=args.similarity_threshold)
        print(f"✅ Built graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        
        # 4. 主题分析
        print(f"\n📊 Step 4: Analyzing topics...")
        analysis_results = builder.analyze_topics()
        print(f"✅ Analyzed {len(analysis_results['topic_distribution'])} topics")
        
        # 5. 生成报告
        print(f"\n📋 Step 5: Generating production report...")
        report = builder.generate_summary_report()
        print("✅ Production report generated")
        
        print(f"\n🎉 Production pipeline completed!")
        print(f"📁 Results: {args.output_dir}")
        print(f"💾 Memory used: {builder._check_memory_usage():.2f} GB")
        
        # 显示关键统计
        print(f"\n📈 Production Stats:")
        print(f"  - Total courses: {len(courses)}")
        print(f"  - Embedding dims: {embeddings.shape[1]}")
        print(f"  - Graph edges: {len(knowledge_graph.edges)}")
        print(f"  - Processing batches: {(len(courses) + args.batch_size - 1) // args.batch_size}")
        
    except Exception as e:
        logger.error(f"生产环境执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
>>>>>>> 700f4dc (优化结构)

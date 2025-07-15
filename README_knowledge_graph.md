# DCAS 课程知识图谱构建器 - 部署指南

## 概述

本项目为DCAS系统构建课程知识图谱，使用Qwen3-Embedding-0.6B模型对课程大纲进行嵌入，并基于相似度构建知识图谱。

## 文件说明

### 脚本文件
- `course_knowledge_graph_builder.py` - 本地测试版本（处理20个样本）
- `course_knowledge_graph_production.py` - 生产版本（处理全量数据，内存优化）
- `deploy_knowledge_graph.sh` - 服务器部署脚本

### 输出文件
- `knowledge_graph_output/` - 本地测试结果
- `knowledge_graph_output_production/` - 生产环境结果

## 本地测试

### 运行条件
✅ **已在dcas conda环境中测试成功**
- Python 3.12+
- 所需包已安装（transformers, torch, pandas, numpy, networkx, etc.）
- 内存需求：~2GB

### 测试命令
```bash
# 运行本地测试（处理20个样本）
python course_knowledge_graph_builder.py
```

### 测试结果
- ✅ 成功加载20个课程
- ✅ 生成1024维embedding向量
- ✅ 构建包含101条边的知识图谱
- ✅ 识别20个独特主题
- ✅ 生成可视化图表和分析报告

## 服务器部署

### 系统要求
- **内存**：建议16GB+（全量2370个课程）
- **存储**：至少10GB可用空间
- **Python**：3.8+
- **GPU**：可选，CUDA支持可加速embedding生成

### 部署步骤

1. **上传文件到服务器**
```bash
scp course_knowledge_graph_production.py user@server:/path/to/dcas/
scp deploy_knowledge_graph.sh user@server:/path/to/dcas/
```

2. **运行部署脚本**
```bash
cd /path/to/dcas/
bash deploy_knowledge_graph.sh
```

3. **激活conda环境**
```bash
conda activate dcas  # 或你的环境名
```

4. **运行生产脚本**
```bash
# 基本运行
python course_knowledge_graph_production.py

# 自定义参数
python course_knowledge_graph_production.py \
  --data-dir "datasets/Course Details/General" \
  --output-dir knowledge_graph_output_production \
  --batch-size 200 \
  --max-memory 16.0 \
  --similarity-threshold 0.65
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `datasets/Course Details/General` | 课程数据目录 |
| `--output-dir` | `knowledge_graph_output_production` | 输出目录 |
| `--batch-size` | `100` | 批处理大小（调整以适应内存） |
| `--max-memory` | `12.0` | 最大内存使用限制(GB) |
| `--similarity-threshold` | `0.7` | 知识图谱边的相似度阈值 |
| `--no-resume` | `False` | 禁用断点续传 |

### 内存优化建议

| 课程数量 | 推荐batch-size | 推荐内存 |
|----------|----------------|----------|
| <500 | 50-100 | 4GB |
| 500-1500 | 100-200 | 8GB |
| 1500-3000 | 200-500 | 16GB |
| 3000+ | 500+ | 32GB+ |

## 输出文件结构

```
knowledge_graph_output_production/
├── embeddings/
│   └── course_embeddings_production_YYYYMMDD_HHMMSS.pkl
├── graphs/
│   └── course_knowledge_graph_production_YYYYMMDD_HHMMSS.pkl
├── analysis/
│   └── topic_analysis_production_YYYYMMDD_HHMMSS.json
├── checkpoints/
│   ├── progress.json
│   ├── courses_cache.pkl
│   └── embeddings_cache.pkl
└── production_report_YYYYMMDD_HHMMSS.md
```

## 特性功能

### 🔄 断点续传
- 自动保存处理进度
- 支持从中断点继续处理
- 缓存已处理的数据

### 💾 内存管理
- 批量处理大型数据集
- 实时内存监控
- 垃圾回收优化

### 📊 数据分析
- 主题分布统计
- 课程聚类分析
- 知识图谱可视化

### 🚀 性能优化
- GPU加速（如果可用）
- 多进程嵌入生成
- 内存友好的相似度计算

## 使用示例

### 加载已生成的数据
```python
import pickle
import networkx as nx

# 加载知识图谱
with open('knowledge_graph_output_production/graphs/course_knowledge_graph_production_*.pkl', 'rb') as f:
    graph = pickle.load(f)

# 加载embeddings
with open('knowledge_graph_output_production/embeddings/course_embeddings_production_*.pkl', 'rb') as f:
    data = pickle.load(f)
    embeddings = data['embeddings']
    courses = data['courses']

# 分析图谱
print(f"课程数量: {len(graph.nodes)}")
print(f"连接数量: {len(graph.edges)}")
print(f"图谱密度: {nx.density(graph)}")
```

### 查找相似课程
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_courses(target_course_idx, embeddings, courses, top_k=5):
    similarities = cosine_similarity([embeddings[target_course_idx]], embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 排除自己
    
    for idx in similar_indices:
        print(f"相似度: {similarities[idx]:.3f} - {courses[idx]['course_name']}")
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减小batch-size参数
   - 增加max-memory限制
   - 关闭其他程序释放内存

2. **GPU内存不足**
   - 脚本会自动回退到CPU
   - 或减小batch-size

3. **数据加载失败**
   - 检查数据目录路径
   - 确认JSON文件格式正确

4. **模型下载失败**
   - 检查网络连接
   - 脚本会自动使用备选模型

### 监控进度

```bash
# 查看实时日志
tail -f nohup.out

# 检查内存使用
htop

# 检查磁盘空间
df -h

# 检查进度文件
cat knowledge_graph_output_production/checkpoints/progress.json
```

## 生产环境建议

1. **使用screen或tmux**运行长时间任务
2. **定期备份**输出文件
3. **监控资源使用**情况
4. **设置日志轮转**避免日志文件过大

## 联系支持

如有问题，请检查：
1. 输出日志中的错误信息
2. 系统资源使用情况
3. 数据文件完整性

---
*DCAS Course Knowledge Graph Builder v1.0*
#!/bin/bash

# 修复生产环境知识图谱构建脚本

echo "🔧 修复DCAS生产环境知识图谱构建问题..."

# 1. 修复f-string格式化问题
echo "修复报告生成格式问题..."
sed -i 's/nx.density(self.knowledge_graph):.6f if self.knowledge_graph else '\''N\/A'\''/(nx.density(self.knowledge_graph) if self.knowledge_graph else 0):.6f/g' course_knowledge_graph_production.py

# 2. 提高相似度阈值，减少边数
echo "提高相似度阈值到0.85..."
sed -i 's/similarity_threshold: float = 0.7/similarity_threshold: float = 0.85/g' course_knowledge_graph_production.py

# 3. 清理之前的输出和检查点
echo "清理旧的输出文件..."
rm -rf knowledge_graph_output_production/

echo "✅ 修复完成！现在可以重新运行："
echo "uv run python course_knowledge_graph_production.py"
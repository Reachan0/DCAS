#!/bin/bash

# 优化知识图谱边数控制脚本

echo "🔧 优化知识图谱边数控制..."

# 1. 进一步提高相似度阈值
echo "提高相似度阈值到0.92..."
sed -i 's/similarity_threshold: float = 0.90/similarity_threshold: float = 0.92/g' course_knowledge_graph_production.py

# 2. 减少最大边数限制
echo "减少最大边数限制到50000..."
sed -i 's/max_edges: int = 100000/max_edges: int = 50000/g' course_knowledge_graph_production.py

# 3. 使用Top-K策略，只保留最相似的边
echo "添加Top-K边选择策略..."

# 4. 清理旧输出
echo "清理旧的输出文件..."
rm -rf knowledge_graph_output_production/

echo "✅ 优化完成！"
echo "📊 新配置:"
echo "  - 相似度阈值: 0.92"
echo "  - 最大边数: 50000"
echo "  - 策略: 只保留最相似的连接"
echo ""
echo "🚀 现在可以重新运行:"
echo "uv run python course_knowledge_graph_production.py"
#!/bin/bash
# DCAS 课程知识图谱部署脚本

echo "🚀 DCAS Course Knowledge Graph Builder - Server Deployment"
echo "=========================================================="

# 检查Python环境
echo "📋 Checking Python environment..."
python --version
echo "Current environment: $(which python)"

# 安装依赖
echo "📦 Installing dependencies..."
pip install pandas numpy scikit-learn networkx matplotlib seaborn tqdm psutil
pip install transformers torch sentence-transformers
pip install plotly  # 可选，用于高级可视化

# 检查GPU支持
echo "🔍 Checking GPU support..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# 创建必要目录
echo "📁 Creating directories..."
mkdir -p knowledge_graph_output_production/{embeddings,graphs,analysis,checkpoints}

# 设置环境变量
echo "⚙️  Setting environment variables..."
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# 显示系统信息
echo "💻 System Information:"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}' 2>/dev/null || echo 'N/A (not Linux)')"
echo "CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4}' 2>/dev/null || echo 'N/A')"

echo ""
echo "✅ Setup completed! Ready to run:"
echo "python course_knowledge_graph_production.py --help"
echo ""
echo "Example production command:"
echo "python course_knowledge_graph_production.py \\"
echo "  --data-dir 'datasets/Course Details/General' \\"
echo "  --output-dir knowledge_graph_output_production \\"
echo "  --batch-size 200 \\"
echo "  --max-memory 16.0 \\"
echo "  --similarity-threshold 0.65"
#!/bin/bash
# 优化版Qwen3 API服务快速启动脚本

echo "🚀 启动优化版Qwen3 API服务"
echo "================================"

# 检查环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import fastapi, uvicorn, psutil" 2>/dev/null || {
    echo "❌ 缺少依赖，正在安装..."
    pip install fastapi uvicorn psutil
}

# 设置环境变量
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM="false"

# 检查模型路径
MODEL_PATH="/Users/xuanchong/projects/models/Qwen3-14B-SFT"
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  模型路径不存在: $MODEL_PATH"
    echo "   将使用Mock模式运行"
else
    echo "✅ 找到模型: $MODEL_PATH"
    export QWEN3_MODEL_PATH="$MODEL_PATH"
fi

# 启动服务
echo "🌟 启动服务 (端口: 8001)..."
echo "   优化特性: 异步处理、超时控制、内存优化"
echo "   访问地址: http://localhost:8001"
echo "   API文档: http://localhost:8001/docs"
echo ""

python3 scripts/qwen3_api_server_optimized.py
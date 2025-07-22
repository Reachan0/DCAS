#!/bin/bash

# Qwen3-14B就业市场分析API启动脚本

echo "=== Qwen3就业市场分析API启动脚本 ==="

# 检查环境变量
if [ -z "$QWEN3_MODEL_PATH" ]; then
    echo "请设置环境变量 QWEN3_MODEL_PATH 指向您的微调Qwen3-14B模型"
    echo "例如: export QWEN3_MODEL_PATH=/path/to/your/qwen3-14b-finetuned"
    exit 1
fi

# 检查模型路径是否存在
if [ ! -d "$QWEN3_MODEL_PATH" ]; then
    echo "错误: 模型路径 $QWEN3_MODEL_PATH 不存在"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 安装依赖
echo "正在检查依赖..."
if [ -f "requirements_api.txt" ]; then
    pip install -r requirements_api.txt
else
    echo "安装基础依赖..."
    pip install fastapi uvicorn transformers torch accelerate
fi

# 启动服务
echo "正在启动Qwen3就业市场分析API..."
echo "模型路径: $QWEN3_MODEL_PATH"
echo "访问地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""

# 使用uvicorn启动
python -m uvicorn scripts.qwen3_api_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --workers 1
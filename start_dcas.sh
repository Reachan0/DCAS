#!/bin/bash
# DCAS简化版启动脚本

echo "🎓 DCAS动态课程对齐系统 - 简化版"
echo "=================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python"
    exit 1
fi

echo "✅ Python环境检查通过"

# 创建虚拟环境 (可选)
if [ "$1" = "--venv" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv dcas_env
    source dcas_env/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 安装依赖
echo "📦 检查并安装依赖..."

# 基础依赖列表
DEPS=(
    "streamlit>=1.28.0"
    "pandas>=1.5.0"
    "numpy>=1.21.0"
    "scikit-learn>=1.1.0"
    "networkx>=2.8.0"
    "matplotlib>=3.5.0"
    "seaborn>=0.11.0"
    "tqdm>=4.64.0"
)

# 安装每个依赖
for dep in "${DEPS[@]}"; do
    echo "安装 $dep..."
    pip install "$dep" --quiet
done

echo "✅ 依赖安装完成"

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p dcas_data
mkdir -p dcas_temp
mkdir -p dcas_output
mkdir -p learner_profiles
mkdir -p alignment_strategies
mkdir -p generated_content
mkdir -p simulation_results
mkdir -p content_templates

echo "✅ 目录创建完成"

# 检查核心模块是否存在
echo "🔍 检查核心模块..."

CORE_FILES=(
    "dcas_core/data_models.py"
    "dcas_core/config.py"
    "dcas_core/content_generator.py"
    "dcas_core/orchestrator.py"
    "dcas_web_app.py"
    "demo.py"
)

missing_files=()
for file in "${CORE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ 错误: 缺少以下核心文件:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo "请确保所有文件都已正确创建"
    exit 1
fi

echo "✅ 核心模块检查通过"

# 运行系统检查
echo "🧪 运行系统检查..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from dcas_core.config import get_config
    from dcas_core.orchestrator import DCASOrchestrator
    
    config = get_config()
    print(f'✅ 配置系统正常 - 版本: {config.get(\"system.version\")}')
    
    # 简单初始化测试
    orchestrator = DCASOrchestrator()
    status = orchestrator.get_system_status()
    print(f'✅ 系统协调器正常 - 状态: {status[\"status\"]}')
    
    print('🎉 系统检查通过！')
except Exception as e:
    print(f'❌ 系统检查失败: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 系统检查失败，请检查错误信息"
    exit 1
fi

# 显示启动选项
echo ""
echo "🚀 DCAS系统已就绪！请选择启动方式:"
echo ""
echo "1. 🖥️  启动Web界面演示:"
echo "   streamlit run dcas_web_app.py"
echo ""
echo "2. 💻 运行命令行演示:"
echo "   python3 demo.py"
echo ""
echo "3. 🧪 快速测试系统:"
echo "   python3 -c \"from dcas_core.orchestrator import DCASOrchestrator; print('✅ 系统正常')\""
echo ""

# 询问是否直接启动
read -p "是否直接启动Web界面？(y/N): " choice
case "$choice" in 
    y|Y|yes|YES ) 
        echo "🌐 启动Web界面..."
        streamlit run dcas_web_app.py
        ;;
    * ) 
        echo "👋 启动脚本完成，请手动选择启动方式"
        ;;
esac
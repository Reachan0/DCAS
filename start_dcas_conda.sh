#!/bin/bash
# DCAS简化版启动脚本 - Conda环境版本

echo "🎓 DCAS动态课程对齐系统 - 简化版 (Conda环境)"
echo "================================================"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "✅ Conda环境检查通过"

# 激活或创建dcas环境
if conda info --envs | grep -q "dcas"; then
    echo "🔄 激活现有的dcas环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dcas
    echo "✅ dcas环境已激活"
else
    echo "🆕 创建新的dcas环境..."
    conda create -n dcas python=3.9 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dcas
    echo "✅ dcas环境创建并激活成功"
fi

# 验证Python环境
python_version=$(python --version 2>&1)
echo "🐍 当前Python版本: $python_version"

# 安装依赖
echo "📦 安装/更新依赖包..."

# 使用conda安装科学计算库 (更稳定)
conda install -y pandas numpy scikit-learn matplotlib seaborn networkx -c conda-forge

# 使用pip安装其他依赖
pip install streamlit>=1.28.0 tqdm>=4.64.0

echo "✅ 依赖安装完成"

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p dcas_data dcas_temp dcas_output learner_profiles alignment_strategies generated_content simulation_results content_templates

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

# 修复Python路径问题
export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "🔧 设置Python路径: $PYTHONPATH"

# 运行系统检查
echo "🧪 运行系统检查..."
python3 -c "
import sys
import os
sys.path.insert(0, '.')

print('🔍 Python路径:', sys.path[:3])
print('📁 当前目录:', os.getcwd())

try:
    from dcas_core.config import get_config
    print('✅ 配置模块导入成功')
    
    config = get_config()
    print(f'✅ 配置系统正常 - 版本: {config.get(\"system.version\")}')
    
    from dcas_core.orchestrator import DCASOrchestrator
    print('✅ 协调器模块导入成功')
    
    # 简单初始化测试
    orchestrator = DCASOrchestrator()
    status = orchestrator.get_system_status()
    print(f'✅ 系统协调器正常 - 状态: {status[\"status\"]}')
    
    print('🎉 系统检查通过！')
except ImportError as e:
    print(f'❌ 导入错误: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
except Exception as e:
    print(f'❌ 系统检查失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 系统检查失败，请检查错误信息"
    exit 1
fi

echo ""
echo "🚀 DCAS系统已就绪！请选择启动方式:"
echo ""
echo "1. 🖥️  启动Web界面演示:"
echo "   streamlit run dcas_web_app.py"
echo ""
echo "2. 💻 运行命令行演示:"
echo "   python3 demo.py"
echo ""
echo "3. 🧪 运行调试测试:"
echo "   python3 debug_test.py"
echo ""

# 询问是否直接启动
read -p "是否直接启动Web界面？(y/N): " choice
case "$choice" in 
    y|Y|yes|YES ) 
        echo "🌐 启动Web界面..."
        streamlit run dcas_web_app.py --server.port 8501 --server.address localhost
        ;;
    * ) 
        echo "👋 启动脚本完成，请手动选择启动方式"
        echo "💡 记住要先运行: conda activate dcas"
        ;;
esac
#!/usr/bin/env python3
"""
环境验证脚本
验证DCAS知识图谱系统的依赖是否正确安装
"""

import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        if import_name == 'PIL':
            import PIL
        elif import_name == 'sklearn':
            import sklearn
        else:
            __import__(import_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - MISSING")
        return False

def check_gpu_support():
    """检查GPU支持"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU支持 - {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("ℹ️  CUDA GPU - 不可用 (将使用CPU)")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_mlx_support():
    """检查MLX支持 (Mac M1/M2)"""
    try:
        import mlx
        print("✅ MLX支持 - 可用")
        return True
    except ImportError:
        print("ℹ️  MLX - 不可用 (仅Mac M1/M2支持)")
        return False

def get_package_version(package_name):
    """获取包版本"""
    try:
        package = __import__(package_name)
        return getattr(package, '__version__', 'unknown')
    except ImportError:
        return 'not installed'

def main():
    """主函数"""
    print("🔍 DCAS知识图谱系统环境检查")
    print("=" * 50)
    
    # 必需的包
    required_packages = [
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('networkx', 'networkx'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('streamlit', 'streamlit'),
        ('psutil', 'psutil'),
        ('tqdm', 'tqdm'),
        ('seaborn', 'seaborn'),
        ('pillow', 'PIL'),
    ]
    
    print("\n📦 核心依赖检查:")
    all_ok = True
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    print("\n🔧 硬件支持检查:")
    check_gpu_support()
    check_mlx_support()
    
    print("\n📊 关键包版本:")
    key_packages = ['transformers', 'torch', 'numpy', 'streamlit']
    for package in key_packages:
        version = get_package_version(package)
        print(f"  {package}: {version}")
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✅ 环境检查通过! 所有依赖已正确安装.")
        print("\n🚀 可以开始使用DCAS知识图谱系统:")
        print("  测试版本: python course_knowledge_graph_builder.py")
        print("  生产版本: python course_knowledge_graph_production.py")
        print("  Web界面: streamlit run streamlit_dashboard.py")
    else:
        print("❌ 环境检查失败! 部分依赖缺失.")
        print("\n🔧 修复建议:")
        print("  运行: uv sync")
        print("  或者: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
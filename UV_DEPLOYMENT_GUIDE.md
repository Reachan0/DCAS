# DCAS Knowledge Graph UV Environment Setup

## 🚀 快速部署指南

### 1. 使用UV恢复环境

```bash
# 安装uv (如果还没有安装)
pip install uv

# 克隆项目
git clone <repository-url>
cd DCAS

# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 2. 可选依赖安装

```bash
# GPU支持 (如果有CUDA)
uv add torch --index-url https://download.pytorch.org/whl/cu118

# MLX支持 (Mac M1/M2)
uv add mlx mlx-lm

# 开发工具
uv add --dev pytest black isort flake8 jupyter
```

### 3. 运行脚本

```bash
# 激活环境
source .venv/bin/activate

# 测试版本(20个样本)
python course_knowledge_graph_builder.py

# 生产版本(完整数据集)
python course_knowledge_graph_production.py

# 启动Web界面
streamlit run streamlit_dashboard.py --server.port 8501
```

## 📋 环境配置详情

### 核心依赖
- **transformers**: 4.53.2+ (Hugging Face模型)
- **torch**: 2.7.1+ (深度学习框架)
- **numpy**: 2.3.1+ (数值计算)
- **pandas**: 2.3.1+ (数据处理)
- **networkx**: 3.5+ (图处理)
- **scikit-learn**: 1.7.0+ (机器学习)
- **matplotlib**: 3.10.3+ (绘图)
- **plotly**: 6.2.0+ (交互式图表)
- **streamlit**: 1.46.1+ (Web界面)
- **psutil**: 7.0.0+ (系统监控)

### 硬件要求
- **最低配置**: 8GB RAM, 4GB存储
- **推荐配置**: 16GB RAM, 8GB存储
- **GPU加速**: 4GB+ VRAM (可选)

## 🔧 故障排除

### 常见问题

1. **UV安装失败**
   ```bash
   # 使用pip安装
   pip install uv
   
   # 或者使用conda
   conda install -c conda-forge uv
   ```

2. **CUDA不可用**
   ```bash
   # 检查CUDA版本
   nvidia-smi
   
   # 安装对应版本的PyTorch
   uv add torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. **内存不足**
   ```bash
   # 减少批处理大小
   # 在脚本中修改: batch_size = 5
   ```

4. **模型下载失败**
   ```bash
   # 设置Hugging Face镜像
   export HF_ENDPOINT=https://hf-mirror.com
   
   # 或使用离线模式
   export TRANSFORMERS_OFFLINE=1
   ```

## 📱 容器化部署

### Docker部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装uv
RUN pip install uv

# 复制配置文件
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["uv", "run", "streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 部署脚本

```bash
# build-docker.sh
#!/bin/bash
docker build -t dcas-knowledge-graph .
docker run -p 8501:8501 -v $(pwd)/datasets:/app/datasets dcas-knowledge-graph
```

## 🌐 服务器部署

### 系统服务配置

```ini
# /etc/systemd/system/dcas-dashboard.service
[Unit]
Description=DCAS Knowledge Graph Dashboard
After=network.target

[Service]
Type=simple
User=dcas
WorkingDirectory=/path/to/DCAS
Environment=PATH=/path/to/DCAS/.venv/bin
ExecStart=/path/to/DCAS/.venv/bin/streamlit run streamlit_dashboard.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx反向代理

```nginx
# /etc/nginx/sites-available/dcas-dashboard
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔍 环境验证

### 验证脚本

```python
# verify_environment.py
import sys
import subprocess

def check_package(package_name):
    try:
        __import__(package_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - MISSING")
        return False

def main():
    required_packages = [
        'transformers', 'torch', 'numpy', 'pandas', 
        'networkx', 'sklearn', 'matplotlib', 'plotly', 
        'streamlit', 'psutil', 'tqdm', 'seaborn', 'PIL'
    ]
    
    print("🔍 环境依赖检查:")
    all_ok = True
    
    for package in required_packages:
        if not check_package(package):
            all_ok = False
    
    if all_ok:
        print("\n✅ 所有依赖已正确安装!")
    else:
        print("\n❌ 部分依赖缺失，请运行: uv sync")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📊 环境监控

### 资源使用监控

```python
# monitor_resources.py
import psutil
import time

def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU: {cpu_percent}% | Memory: {memory.percent}%")
        time.sleep(5)

if __name__ == "__main__":
    monitor_resources()
```

---

## 🎯 总结

通过UV包管理器，您可以:

1. **快速环境复制**: 使用`uv sync`一键恢复环境
2. **依赖锁定**: 确保所有环境使用相同版本
3. **高效安装**: UV比pip快10-100倍
4. **完整隔离**: 每个项目独立的虚拟环境

现在您可以轻松在任何机器上部署DCAS知识图谱系统!
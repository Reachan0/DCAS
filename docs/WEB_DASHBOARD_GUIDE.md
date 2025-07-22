# DCAS 课程知识图谱可视化界面部署指南

## 🌐 Web界面概述

我已经为您创建了一个基于Streamlit的交互式Web可视化界面，可以通过浏览器方便地探索和分析课程知识图谱数据。

## ✅ 部署状态

### 已完成的功能
- ✅ **本地部署成功** - Streamlit应用已启动在 http://localhost:8501
- ✅ **数据加载完成** - 自动检测并加载测试数据
- ✅ **多页面界面** - 6个功能模块完整实现
- ✅ **交互式可视化** - 基于Plotly的动态图表

### 界面功能模块

#### 🏠 首页概览
- 数据集统计指标展示
- 热门主题分布图
- 快速搜索功能
- 系统状态概览

#### 🔍 课程搜索
- **基础搜索**: 关键词匹配
- **高级搜索**: 多条件过滤
- **权重调节**: 自定义搜索权重
- **结果展示**: 评分排序和详情查看

#### 🕸️ 知识图谱
- **交互式网络图**: 课程关联关系可视化
- **布局算法**: 多种图形布局选择
- **节点控制**: 节点数量和大小调节
- **主题过滤**: 按主题高亮显示
- **中心性分析**: 度中心性和介数中心性

#### 📊 主题分析
- **主题分布**: 热门主题统计
- **聚类分析**: 课程自动聚类
- **相关性矩阵**: 主题间关联度
- **可视化图表**: 柱状图、饼图、热力图

#### 🎯 相似度分析
- **课程选择**: 搜索并选择目标课程
- **相似度计算**: 基于embedding的余弦相似度
- **阈值过滤**: 可调节相似度阈值
- **降维可视化**: PCA/t-SNE可视化
- **主题重叠**: 分析相似课程的主题交集

#### 📈 统计报告
- **数据质量报告**: 完整性统计
- **分布分析**: 课程描述长度、主题数量分布
- **图谱拓扑**: 度分布、连通分量分析
- **数据导出**: CSV/JSON格式下载

## 🚀 快速启动

### 1. 本地访问（已启动）
```bash
# 应用已在运行
# 访问地址: http://localhost:8501
```

### 2. 服务器部署
```bash
# 运行部署脚本
./deploy_dashboard.sh

# 启动服务器版本（允许外部访问）
./start_dashboard_server.sh

# 访问地址: http://YOUR_SERVER_IP:8501
```

### 3. 手动启动
```bash
# 基本启动
streamlit run streamlit_dashboard.py

# 服务器模式
streamlit run streamlit_dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

## 🎯 使用演示

### 搜索功能演示
1. **关键词搜索**:
   - 在首页搜索框输入 "Programming"
   - 查看匹配结果和相关度评分

2. **高级搜索**:
   - 进入搜索页面
   - 选择主题过滤器
   - 调整权重参数
   - 执行多条件搜索

### 知识图谱探索
1. **图谱可视化**:
   - 调整显示节点数量（建议50个）
   - 选择布局算法（spring效果最好）
   - 按主题筛选显示

2. **中心性分析**:
   - 查看度中心性最高的课程
   - 分析关键节点的连接模式

### 相似度分析
1. **选择目标课程**:
   - 搜索感兴趣的课程
   - 设置相似度阈值
   - 查看相似课程列表

2. **可视化分析**:
   - 使用PCA或t-SNE降维
   - 观察课程在向量空间中的分布

## 📊 数据接口

### 数据加载机制
- **自动检测**: 优先加载生产数据，回退到测试数据
- **缓存机制**: 使用Streamlit缓存提升加载性能
- **增量更新**: 支持数据更新后自动刷新

### 支持的数据格式
- **知识图谱**: NetworkX pickle格式 (.pkl)
- **向量数据**: NumPy数组 + 课程元数据
- **分析结果**: JSON格式主题和聚类数据

## 🔧 自定义配置

### 修改端口
```bash
streamlit run streamlit_dashboard.py --server.port 8080
```

### 配置文件
创建 `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

## 🔒 安全考虑

### 本地部署
- 默认只允许本地访问（127.0.0.1）
- 适合开发和测试环境

### 服务器部署
- 配置防火墙规则限制访问
- 考虑使用反向代理（Nginx/Apache）
- 添加身份验证机制

### 推荐配置
```bash
# 防火墙配置示例
sudo ufw allow from YOUR_IP_RANGE to any port 8501

# Nginx反向代理配置
location /dashboard {
    proxy_pass http://localhost:8501;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## 📱 移动端支持

### 响应式设计
- 界面自适应不同屏幕尺寸
- 支持平板和手机访问
- 触摸友好的交互控件

### 性能优化
- 数据缓存减少加载时间
- 图表懒加载提升响应速度
- 批量数据处理优化内存使用

## 🛠️ 故障排除

### 常见问题

1. **端口被占用**
```bash
# 查看端口使用情况
lsof -i :8501

# 杀掉占用进程
kill -9 PID

# 或使用其他端口
streamlit run streamlit_dashboard.py --server.port 8502
```

2. **数据加载失败**
```bash
# 检查数据文件是否存在
ls -la knowledge_graph_output*/

# 重新生成数据
python course_knowledge_graph_builder.py
```

3. **内存不足**
```bash
# 减少图谱显示节点数
# 在界面中调整max_nodes参数为较小值（如20-30）

# 监控内存使用
top -p $(pgrep -f streamlit)
```

4. **网络访问问题**
```bash
# 检查防火墙设置
sudo ufw status

# 测试端口连通性
telnet YOUR_SERVER_IP 8501
```

## 📈 性能监控

### 实时监控
```bash
# 监控应用进程
ps aux | grep streamlit

# 监控内存和CPU使用
htop -p $(pgrep -f streamlit)

# 监控网络连接
netstat -an | grep 8501
```

### 日志查看
```bash
# Streamlit自动日志
tail -f ~/.streamlit/logs/streamlit.log

# 自定义日志记录
# 在应用中添加logging配置
```

## 🚀 扩展功能

### 建议的增强功能
1. **用户认证系统**
2. **数据更新定时任务**
3. **多语言支持**
4. **高级分析工具**
5. **数据导出API**

### 集成建议
- **数据库集成**: 连接PostgreSQL/MongoDB
- **API接口**: 提供RESTful API
- **云部署**: 部署到AWS/Azure/GCP
- **容器化**: Docker部署方案

---

## 🎉 部署完成总结

✅ **Web界面已成功部署**
- 本地访问: http://localhost:8501
- 功能完整: 6个主要模块全部可用
- 数据加载: 自动检测并加载知识图谱数据
- 交互体验: 现代化的Web界面，支持实时交互

✅ **核心功能验证**
- 课程搜索和过滤 ✓
- 知识图谱可视化 ✓
- 主题分析和聚类 ✓
- 相似度计算和可视化 ✓
- 统计报告和数据导出 ✓

🚀 **立即可用**
界面已完全就绪，您可以立即开始使用Web界面探索课程知识图谱数据！
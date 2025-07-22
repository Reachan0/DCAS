#!/bin/bash
# DCAS 课程知识图谱可视化界面部署脚本

echo "🌐 DCAS Course Knowledge Graph Web Dashboard"
echo "============================================="

# 检查是否已安装Streamlit
echo "📋 Checking dependencies..."
python -c "import streamlit, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing web dependencies..."
    pip install streamlit plotly
else
    echo "✅ Web dependencies already installed"
fi

# 检查数据文件
echo "📁 Checking data files..."
if [ -d "knowledge_graph_output" ] || [ -d "knowledge_graph_output_production" ]; then
    echo "✅ Knowledge graph data found"
else
    echo "❌ No knowledge graph data found"
    echo "Please run the knowledge graph builder first:"
    echo "  python course_knowledge_graph_builder.py  # for test data"
    echo "  python course_knowledge_graph_production.py  # for full data"
    exit 1
fi

# 创建启动脚本
cat > start_dashboard.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting DCAS Knowledge Graph Dashboard..."
echo "Access the dashboard at: http://localhost:8501"
echo "Press Ctrl+C to stop"
streamlit run streamlit_dashboard.py --server.port 8501
EOF

chmod +x start_dashboard.sh

# 创建服务器部署脚本
cat > start_dashboard_server.sh << 'EOF'
#!/bin/bash
echo "🌐 Starting DCAS Dashboard for Server Deployment..."
echo "Dashboard will be accessible on all network interfaces"

# 获取服务器IP
SERVER_IP=$(hostname -I | cut -d' ' -f1 2>/dev/null || ifconfig | grep "inet " | grep -v 127.0.0.1 | cut -d' ' -f2 | head -1)

echo "Access URLs:"
echo "  Local: http://localhost:8501"
echo "  Network: http://${SERVER_IP}:8501"
echo ""
echo "Press Ctrl+C to stop the server"

# 启动服务器版本（允许外部访问）
streamlit run streamlit_dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
EOF

chmod +x start_dashboard_server.sh

echo ""
echo "✅ Dashboard deployment completed!"
echo ""
echo "📖 Usage Instructions:"
echo "  For local access:"
echo "    ./start_dashboard.sh"
echo ""
echo "  For server deployment (external access):"
echo "    ./start_dashboard_server.sh"
echo ""
echo "  Or run directly:"
echo "    streamlit run streamlit_dashboard.py"
echo ""
echo "🌐 Dashboard Features:"
echo "  • 🏠 Home Overview - Data statistics and quick search"
echo "  • 🔍 Course Search - Advanced search with filters"
echo "  • 🕸️ Knowledge Graph - Interactive graph visualization"
echo "  • 📊 Topic Analysis - Topic distribution and clustering"
echo "  • 🎯 Similarity Analysis - Course similarity exploration"
echo "  • 📈 Statistics Report - Comprehensive data analysis"
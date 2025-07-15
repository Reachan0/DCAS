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

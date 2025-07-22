#!/bin/bash
echo "🚀 Starting DCAS Knowledge Graph Dashboard..."
echo "Access the dashboard at: http://localhost:8501"
echo "Press Ctrl+C to stop"
streamlit run streamlit_dashboard.py --server.port 8501

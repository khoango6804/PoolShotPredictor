#!/bin/bash

# YOLOv11 Billiards Detection - Network Access Script
echo "🎱 YOLOv11 Billiards Detection - Network Access"
echo "=============================================="

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "🌐 Local IP: $LOCAL_IP"

# Check if port 8501 is available
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Port 8501 is in use, using port 8502"
    PORT=8502
else
    PORT=8501
fi

echo "🚀 Starting web interface on port $PORT"
echo "📱 Local URL: http://localhost:$PORT"
echo "🌐 Network URL: http://$LOCAL_IP:$PORT"
echo "⏹️ Press Ctrl+C to stop"
echo "=============================================="

# Start Streamlit with network access
streamlit run web_interface.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false 
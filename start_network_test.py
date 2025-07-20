#!/usr/bin/env python3
"""
🎱 YOLOv11 Billiards Detection - Network Test
==================================================
Simple script to start web interface with network access
"""

import subprocess
import sys
import socket
import time

def get_local_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def check_port(port):
    """Check if port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # True if port is free
    except:
        return False

def main():
    print("🎱 YOLOv11 Billiards Detection - Network Test")
    print("=" * 50)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"🌐 Local IP: {local_ip}")
    
    # Find available port
    ports = [8501, 8502, 8503, 8504, 8505]
    available_port = None
    
    for port in ports:
        if check_port(port):
            available_port = port
            break
    
    if not available_port:
        print("❌ No available ports found!")
        return
    
    print(f"✅ Using port: {available_port}")
    print(f"🌐 Network URL: http://{local_ip}:{available_port}")
    print("=" * 50)
    
    # Start Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "web_interface.py",
        "--server.address", "0.0.0.0",
        "--server.port", str(available_port),
        "--server.headless", "true"
    ]
    
    print("🚀 Starting web interface...")
    print(f"📱 Local URL: http://localhost:{available_port}")
    print(f"🌐 Network URL: http://{local_ip}:{available_port}")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping server...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
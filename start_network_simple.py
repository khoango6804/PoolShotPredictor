#!/usr/bin/env python3
"""
Simple Network Access Script for YOLOv11 Billiards Detection
"""

import subprocess
import socket
import sys
import os

def get_local_ip():
    """Get local IP address"""
    try:
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
            return True
    except:
        return False

def main():
    print("🎱 YOLOv11 Billiards Detection - Network Access")
    print("=" * 50)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"🌐 Local IP: {local_ip}")
    
    # Choose port
    port = 8501
    if not check_port(port):
        port = 8502
        print(f"⚠️ Port 8501 in use, using port {port}")
    
    # Choose interface
    print("\n🎯 Choose interface:")
    print("1. Basic Interface (Simple)")
    print("2. Advanced Interface (Full features)")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        interface_file = "web_interface.py"
        print("🚀 Starting Basic Interface...")
    elif choice == "2":
        interface_file = "web_interface_advanced.py"
        print("🚀 Starting Advanced Interface...")
    else:
        print("❌ Invalid choice")
        return
    
    print(f"📱 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{local_ip}:{port}")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Start Streamlit with network access
        cmd = [
            sys.executable, "-m", "streamlit", "run", interface_file,
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ Interface stopped")

if __name__ == "__main__":
    main() 
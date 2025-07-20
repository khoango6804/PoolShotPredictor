#!/usr/bin/env python3
"""
Network-enabled Web Interface for YOLOv11 Billiards Detection
Allows access from other devices on the network
"""

import subprocess
import sys
import socket
import requests
import time
from pathlib import Path

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

def check_port_available(port):
    """Check if port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
            return True
    except:
        return False

def find_available_port(start_port=8501):
    """Find available port starting from start_port"""
    port = start_port
    while port < start_port + 100:
        if check_port_available(port):
            return port
        port += 1
    return None

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import streamlit
        import ultralytics
        import cv2
        import numpy as np
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_models():
    """Check if models are available"""
    model_paths = [
        "runs/detect/yolo11_billiards_gpu/weights/best.pt",
        "table_detector/yolo11_table_detector/weights/best.pt",
        "pocket_detector/yolo11_pocket_detector/weights/best.pt"
    ]
    
    available_models = []
    for path in model_paths:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            available_models.append(f"✅ {path} ({size_mb:.1f} MB)")
        else:
            available_models.append(f"❌ {path} - Not found")
    
    return available_models

def main():
    print("🎱 YOLOv11 Network Billiards Detection Web Interface")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Please install missing dependencies:")
        print("pip install ultralytics streamlit opencv-python pillow pandas numpy")
        return
    
    print("✅ All dependencies are installed!")
    
    # Check models
    print("\n🔍 Checking models...")
    models = check_models()
    for model in models:
        print(f"  {model}")
    
    # Get network info
    local_ip = get_local_ip()
    print(f"\n🌐 Network Information:")
    print(f"  Local IP: {local_ip}")
    
    # Find available ports
    basic_port = find_available_port(8501)
    advanced_port = find_available_port(8503)
    
    if not basic_port or not advanced_port:
        print("❌ No available ports found")
        return
    
    print(f"  Basic Interface Port: {basic_port}")
    print(f"  Advanced Interface Port: {advanced_port}")
    
    # Display access URLs
    print(f"\n🚀 Access URLs:")
    print(f"  Basic Interface:")
    print(f"    Local: http://localhost:{basic_port}")
    print(f"    Network: http://{local_ip}:{basic_port}")
    print(f"  Advanced Interface:")
    print(f"    Local: http://localhost:{advanced_port}")
    print(f"    Network: http://{local_ip}:{advanced_port}")
    
    # Choose interface
    print(f"\n🎯 Choose interface to start:")
    print(f"  1. Basic Interface (Port {basic_port})")
    print(f"  2. Advanced Interface (Port {advanced_port})")
    print(f"  3. Both interfaces")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        start_basic_interface(basic_port, local_ip)
    elif choice == "2":
        start_advanced_interface(advanced_port, local_ip)
    elif choice == "3":
        start_both_interfaces(basic_port, advanced_port, local_ip)
    else:
        print("❌ Invalid choice")

def start_basic_interface(port, local_ip):
    """Start basic web interface"""
    print(f"\n🚀 Starting Basic Interface...")
    print(f"📱 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{local_ip}:{port}")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Start Streamlit with network access
        cmd = [
            sys.executable, "-m", "streamlit", "run", "web_interface.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ Basic interface stopped")

def start_advanced_interface(port, local_ip):
    """Start advanced web interface"""
    print(f"\n🚀 Starting Advanced Interface...")
    print(f"📱 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{local_ip}:{port}")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Start Streamlit with network access
        cmd = [
            sys.executable, "-m", "streamlit", "run", "web_interface_advanced.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ Advanced interface stopped")

def start_both_interfaces(basic_port, advanced_port, local_ip):
    """Start both interfaces"""
    print(f"\n🚀 Starting Both Interfaces...")
    print(f"📱 Basic: http://localhost:{basic_port} | http://{local_ip}:{basic_port}")
    print(f"📱 Advanced: http://localhost:{advanced_port} | http://{local_ip}:{advanced_port}")
    print("⏹️ Press Ctrl+C to stop all")
    print("=" * 60)
    
    try:
        # Start basic interface in background
        basic_cmd = [
            sys.executable, "-m", "streamlit", "run", "web_interface.py",
            "--server.port", str(basic_port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        basic_process = subprocess.Popen(basic_cmd)
        
        # Wait a bit for basic interface to start
        time.sleep(3)
        
        # Start advanced interface
        advanced_cmd = [
            sys.executable, "-m", "streamlit", "run", "web_interface_advanced.py",
            "--server.port", str(advanced_port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        advanced_process = subprocess.Popen(advanced_cmd)
        
        # Wait for both processes
        try:
            basic_process.wait()
            advanced_process.wait()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping all interfaces...")
            basic_process.terminate()
            advanced_process.terminate()
            
    except Exception as e:
        print(f"❌ Error starting interfaces: {e}")

if __name__ == "__main__":
    main() 
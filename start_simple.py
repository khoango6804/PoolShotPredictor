#!/usr/bin/env python3
"""
🎱 Simple Network Test
"""

import subprocess
import sys
import socket

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    ip = get_ip()
    print(f"🌐 IP: {ip}")
    print(f"🌐 Network URL: http://{ip}:8501")
    print("🚀 Starting...")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "web_interface.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Start YOLOv11 Billiards Detection Web Interface
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import ultralytics
        import cv2
        import numpy
        import PIL
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install -r requirements_web.txt")
        return False

def check_model():
    """Check if YOLOv11 model exists"""
    model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    if Path(model_path).exists():
        print(f"✅ Model found: {model_path}")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the YOLOv11 model is trained and available.")
        return False

def start_web_interface():
    """Start the Streamlit web interface"""
    print("🎱 Starting YOLOv11 Billiards Detection Web Interface...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check model
    if not check_model():
        return False
    
    print("\n🚀 Launching web interface...")
    print("📱 The interface will open in your default web browser")
    print("🌐 Local URL: http://localhost:8501")
    print("🔗 Network URL: http://your-ip:8501")
    print("\n💡 Features:")
    print("  📸 Upload images (JPG, PNG)")
    print("  🎬 Upload videos (MP4, AVI, MOV, MKV)")
    print("  ⚙️ Adjustable confidence threshold")
    print("  📊 Real-time detection statistics")
    print("  📥 Download processed results")
    print("\n⏹️ Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "web_interface.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎱 YOLOv11 Billiards Detection Web Interface")
    print("=" * 50)
    
    # Start the interface
    success = start_web_interface()
    
    if success:
        print("✅ Web interface started successfully!")
    else:
        print("❌ Failed to start web interface")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed:")
        print("   pip install -r requirements_web.txt")
        print("2. Ensure YOLOv11 model is trained:")
        print("   python train_yolo11.py")
        print("3. Check if port 8501 is available")
        print("4. Try running manually:")
        print("   streamlit run web_interface.py")

if __name__ == "__main__":
    main() 
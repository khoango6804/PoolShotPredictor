#!/usr/bin/env python3
"""
Start Advanced YOLOv11 Billiards Detection Web Interface
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
        import pandas
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install streamlit ultralytics opencv-python numpy pillow pandas")
        return False

def check_models():
    """Check if any models are available"""
    models = {
        "Ball Detector": "runs/detect/yolo11_billiards_gpu/weights/best.pt",
        "Table Detector": "table_detector/yolo11_table_detector/weights/best.pt",
        "Pocket Detector": "pocket_detector/yolo11_pocket_detector/weights/best.pt"
    }
    
    available_models = []
    for name, path in models.items():
        if Path(path).exists():
            available_models.append(name)
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - Not found")
    
    return len(available_models) > 0

def start_advanced_interface():
    """Start the advanced Streamlit web interface"""
    print("🎱 Starting YOLOv11 Advanced Billiards Detection Web Interface...")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check models
    if not check_models():
        print("\n⚠️  No models found!")
        print("Please train models first using:")
        print("  python train_all_models.py")
        print("\nOr continue with basic interface...")
    
    print("\n🚀 Launching advanced web interface...")
    print("📱 The interface will open in your default web browser")
    print("🌐 Local URL: http://localhost:8503")
    print("🔗 Network URL: http://your-ip:8503")
    print("\n💡 Advanced Features:")
    print("  📸 Image Detection with advanced parameters")
    print("  🎬 Video Processing with frame control")
    print("  📹 Real-time Detection with camera")
    print("  ⚙️ Adjustable confidence, IoU, max detections")
    print("  🎛️ Frame skip, resize, FPS control")
    print("  📊 Detailed statistics and analysis")
    print("  📥 Download processed results")
    print("\n⏹️ Press Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        # Start Streamlit on different port
        subprocess.run([sys.executable, "-m", "streamlit", "run", "web_interface_advanced.py", "--server.port", "8503"])
    except KeyboardInterrupt:
        print("\n🛑 Advanced web interface stopped by user")
    except Exception as e:
        print(f"❌ Error starting advanced web interface: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎱 YOLOv11 Advanced Billiards Detection Web Interface")
    print("=" * 60)
    
    # Start the interface
    success = start_advanced_interface()
    
    if success:
        print("✅ Advanced web interface started successfully!")
    else:
        print("❌ Failed to start advanced web interface")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed:")
        print("   pip install streamlit ultralytics opencv-python numpy pillow pandas")
        print("2. Ensure at least one model is trained:")
        print("   python train_all_models.py")
        print("3. Check if port 8503 is available")
        print("4. Try running manually:")
        print("   streamlit run web_interface_advanced.py --server.port 8503")

if __name__ == "__main__":
    main() 
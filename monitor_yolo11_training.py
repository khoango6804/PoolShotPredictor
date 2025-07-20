#!/usr/bin/env python3
"""
Monitor YOLOv11 training progress
"""

import os
import time
import json
from pathlib import Path

def monitor_training():
    """Monitor YOLOv11 training progress"""
    
    print("📊 YOLOv11 Training Monitor")
    print("=" * 40)
    
    # Training directory
    train_dir = Path("runs/detect/yolo11_billiards_v1")
    
    if not train_dir.exists():
        print("❌ Training directory not found!")
        return
    
    print(f"📁 Training directory: {train_dir}")
    
    # Check for results file
    results_file = train_dir / "results.csv"
    if results_file.exists():
        print(f"✅ Results file found: {results_file}")
        
        # Read last few lines
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                print(f"📈 Latest results:")
                for line in lines[-3:]:  # Last 3 lines
                    print(f"  {line.strip()}")
    
    # Check for weights
    weights_dir = train_dir / "weights"
    if weights_dir.exists():
        print(f"\n🤖 Weights directory: {weights_dir}")
        
        # List weight files
        weight_files = list(weights_dir.glob("*.pt"))
        if weight_files:
            print(f"📦 Weight files found:")
            for weight_file in weight_files:
                size_mb = weight_file.stat().st_size / (1024 * 1024)
                print(f"  {weight_file.name} ({size_mb:.1f} MB)")
    
    # Check for plots
    plots_dir = train_dir
    plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
    if plot_files:
        print(f"\n📊 Plot files:")
        for plot_file in plot_files:
            print(f"  {plot_file.name}")
    
    # Check dataset info
    print(f"\n📋 Dataset Information:")
    print(f"  Classes: 23")
    print(f"  Train images: 3,905")
    print(f"  Valid images: 414")
    print(f"  Test images: 202")
    print(f"  Total: 4,521 images")
    
    # Check model info
    print(f"\n🤖 Model Information:")
    print(f"  Model: YOLOv11n")
    print(f"  Parameters: ~2.6M")
    print(f"  Device: CPU")
    print(f"  Batch size: 16")
    print(f"  Image size: 640x640")

def check_training_status():
    """Check if training is still running"""
    
    train_dir = Path("runs/detect/yolo11_billiards_v1")
    
    if not train_dir.exists():
        return False
    
    # Check for lock file or active process
    lock_file = train_dir / ".lock"
    if lock_file.exists():
        return True
    
    # Check if results file is being updated
    results_file = train_dir / "results.csv"
    if results_file.exists():
        # Check if file was modified recently (within last 5 minutes)
        mtime = results_file.stat().st_mtime
        if time.time() - mtime < 300:  # 5 minutes
            return True
    
    return False

if __name__ == "__main__":
    monitor_training()
    
    print(f"\n🔄 Training Status:")
    if check_training_status():
        print("✅ Training is active")
    else:
        print("⏸️ Training may have stopped")
    
    print(f"\n💡 Tips:")
    print(f"  - Check 'runs/detect/yolo11_billiards_v1/results.csv' for progress")
    print(f"  - Best model will be saved as 'best.pt'")
    print(f"  - Training plots are saved in the same directory") 
#!/usr/bin/env python3
"""
Monitor YOLOv11 training progress
"""

import time
import os
from pathlib import Path
import subprocess
import sys

def check_training_status():
    """Check if any training is running"""
    print("🔍 Checking training status...")
    
    # Check for training processes
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if 'train_table_detector.py' in result.stdout or 'train_pocket_detector.py' in result.stdout:
            print("✅ Training process detected")
            return True
        else:
            print("❌ No training process detected")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not check process status: {e}")
        return False

def check_training_outputs():
    """Check training output directories"""
    print("\n📁 Checking training outputs...")
    
    training_dirs = [
        "table_detector/yolo11_table_detector",
        "table_detector/yolo11_table_detector_continued", 
        "pocket_detector/yolo11_pocket_detector",
        "pocket_detector/yolo11_pocket_detector_continued"
    ]
    
    for dir_path in training_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
            
            # Check for weights
            weights_path = Path(dir_path) / "weights"
            if weights_path.exists():
                weight_files = list(weights_path.glob("*.pt"))
                if weight_files:
                    latest_weight = max(weight_files, key=lambda x: x.stat().st_mtime)
                    size_mb = latest_weight.stat().st_size / (1024 * 1024)
                    print(f"   📦 Latest weight: {latest_weight.name} ({size_mb:.1f} MB)")
            
            # Check for results
            results_path = Path(dir_path) / "results.csv"
            if results_path.exists():
                print(f"   📊 Results file: {results_path}")
        else:
            print(f"❌ {dir_path} - Not found")

def show_training_commands():
    """Show available training commands"""
    print("\n🚀 Available Training Commands:")
    print("=" * 40)
    print("1. Train Table Detector:")
    print("   python train_table_detector.py")
    print()
    print("2. Train Pocket Detector:")
    print("   python train_pocket_detector.py")
    print()
    print("3. Train Both (Interactive):")
    print("   python train_all_models.py")
    print()
    print("4. Check Model Status:")
    print("   python check_model_status.py")

def main():
    """Main function"""
    print("🎱 YOLOv11 Training Monitor")
    print("=" * 40)
    
    while True:
        print("\n📋 Options:")
        print("1. Check training status")
        print("2. Check training outputs")
        print("3. Show training commands")
        print("4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                check_training_status()
            elif choice == "2":
                check_training_outputs()
            elif choice == "3":
                show_training_commands()
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main() 
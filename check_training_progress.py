#!/usr/bin/env python3
"""
Check YOLOv11 training progress in real-time
"""

import time
import os
from pathlib import Path
import subprocess
import sys

def check_training_directories():
    """Check training output directories"""
    print("🔍 Checking training directories...")
    
    training_dirs = [
        "table_detector/yolo11_table_detector",
        "table_detector/yolo11_table_detector_continued",
        "pocket_detector/yolo11_pocket_detector", 
        "pocket_detector/yolo11_pocket_detector_continued"
    ]
    
    active_training = []
    
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
                    mod_time = time.ctime(latest_weight.stat().st_mtime)
                    print(f"   📦 Latest: {latest_weight.name} ({size_mb:.1f} MB) - {mod_time}")
                    active_training.append(dir_path)
            
            # Check for results
            results_path = Path(dir_path) / "results.csv"
            if results_path.exists():
                print(f"   📊 Results: {results_path}")
                
                # Read last few lines of results
                try:
                    with open(results_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip()
                            print(f"   📈 Last result: {last_line}")
                except:
                    pass
        else:
            print(f"❌ {dir_path} - Not found")
    
    return active_training

def check_processes():
    """Check if training processes are running"""
    print("\n🔍 Checking training processes...")
    
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        training_processes = []
        for line in result.stdout.split('\n'):
            if 'train_table_detector.py' in line or 'train_pocket_detector.py' in line:
                training_processes.append(line.strip())
        
        if training_processes:
            print("✅ Training processes detected:")
            for proc in training_processes:
                print(f"   🔄 {proc}")
            return True
        else:
            print("❌ No training processes detected")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")
        return False

def monitor_training():
    """Monitor training progress continuously"""
    print("🎱 YOLOv11 Training Monitor")
    print("=" * 40)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 40)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🎱 YOLOv11 Training Monitor")
            print("=" * 40)
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 40)
            
            # Check processes
            processes_running = check_processes()
            
            # Check directories
            active_training = check_training_directories()
            
            # Summary
            print(f"\n📊 Summary:")
            print(f"   Processes running: {'Yes' if processes_running else 'No'}")
            print(f"   Active training dirs: {len(active_training)}")
            
            if active_training:
                print(f"   Active: {', '.join([Path(d).name for d in active_training])}")
            
            # Wait before next check
            print(f"\n⏳ Next check in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

def main():
    """Main function"""
    print("🎱 YOLOv11 Training Progress Checker")
    print("=" * 50)
    
    print("📋 Options:")
    print("1. Check current status")
    print("2. Monitor continuously")
    print("3. Exit")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            check_processes()
            check_training_directories()
        elif choice == "2":
            monitor_training()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 
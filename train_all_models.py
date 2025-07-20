#!/usr/bin/env python3
"""
Train all YOLOv11 models: Table Detector and Pocket Detector
"""

import subprocess
import sys
import time
from pathlib import Path

def check_datasets():
    """Check if all datasets are available"""
    print("🔍 Checking datasets...")
    
    datasets = {
        "Table Detector": "table detector/data.yaml",
        "Pocket Detector": "pocket detection/data.yaml"
    }
    
    all_available = True
    for name, path in datasets.items():
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - NOT FOUND")
            all_available = False
    
    return all_available

def train_table_detector():
    """Train table detector"""
    print("\n🎯 Starting Table Detector Training")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "train_table_detector.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Table detector training completed successfully!")
            return True
        else:
            print(f"❌ Table detector training failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running table detector training: {e}")
        return False

def train_pocket_detector():
    """Train pocket detector"""
    print("\n🎯 Starting Pocket Detector Training")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "train_pocket_detector.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pocket detector training completed successfully!")
            return True
        else:
            print(f"❌ Pocket detector training failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running pocket detector training: {e}")
        return False

def show_training_menu():
    """Show training options menu"""
    print("\n🎯 YOLOv11 Training Menu")
    print("=" * 30)
    print("1. Train Table Detector")
    print("2. Train Pocket Detector")
    print("3. Train Both Models")
    print("4. Check Model Status")
    print("5. Exit")
    print("=" * 30)
    
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                return "table"
            elif choice == "2":
                return "pocket"
            elif choice == "3":
                return "both"
            elif choice == "4":
                return "status"
            elif choice == "5":
                return "exit"
            else:
                print("❌ Invalid choice. Please select 1-5.")
        except KeyboardInterrupt:
            return "exit"

def check_model_status():
    """Check status of trained models"""
    print("\n📊 Model Status Check")
    print("=" * 40)
    
    models = {
        "Table Detector": [
            "table_detector/yolo11_table_detector/weights/best.pt",
            "table_detector/yolo11_table_detector_continued/weights/best.pt"
        ],
        "Pocket Detector": [
            "pocket_detector/yolo11_pocket_detector/weights/best.pt",
            "pocket_detector/yolo11_pocket_detector_continued/weights/best.pt"
        ]
    }
    
    for model_name, paths in models.items():
        print(f"\n🎯 {model_name}:")
        for path in paths:
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"  ✅ {path} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {path} - Not found")

def main():
    """Main function"""
    print("🎱 YOLOv11 Multi-Model Training System")
    print("=" * 50)
    
    # Check datasets first
    if not check_datasets():
        print("\n❌ Some datasets are missing!")
        print("Please ensure all datasets are available before training.")
        return
    
    while True:
        choice = show_training_menu()
        
        if choice == "exit":
            print("\n👋 Goodbye!")
            break
        elif choice == "status":
            check_model_status()
        elif choice == "table":
            train_table_detector()
        elif choice == "pocket":
            train_pocket_detector()
        elif choice == "both":
            print("\n🚀 Training both models sequentially...")
            
            # Train table detector first
            if train_table_detector():
                print("\n⏳ Waiting 5 seconds before starting pocket detector...")
                time.sleep(5)
                
                # Train pocket detector
                train_pocket_detector()
            else:
                print("❌ Table detector training failed. Stopping.")
        
        if choice != "status":
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 
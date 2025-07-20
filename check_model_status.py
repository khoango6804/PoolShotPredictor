#!/usr/bin/env python3
"""
Check status of all YOLOv11 models
"""

from pathlib import Path
import os

def check_model_status():
    """Check status of all models"""
    print("🎱 YOLOv11 Model Status Check")
    print("=" * 50)
    
    models = {
        "Ball Detector": [
            "runs/detect/yolo11_billiards_gpu/weights/best.pt",
            "runs/detect/yolo11_billiards_v2/weights/best.pt",
            "runs/detect/yolo11_billiards_v1/weights/best.pt"
        ],
        "Table Detector": [
            "table_detector/yolo11_table_detector/weights/best.pt",
            "table_detector/yolo11_table_detector_continued/weights/best.pt"
        ],
        "Pocket Detector": [
            "pocket_detector/yolo11_pocket_detector/weights/best.pt",
            "pocket_detector/yolo11_pocket_detector_continued/weights/best.pt"
        ]
    }
    
    total_models = 0
    available_models = 0
    
    for model_name, paths in models.items():
        print(f"\n🎯 {model_name}:")
        found = False
        
        for path in paths:
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"  ✅ {path} ({size_mb:.1f} MB)")
                found = True
                available_models += 1
            else:
                print(f"  ❌ {path} - Not found")
            
            total_models += 1
        
        if not found:
            print(f"  ⚠️  No {model_name} models found")
    
    print(f"\n📊 Summary:")
    print(f"  Total model paths checked: {total_models}")
    print(f"  Available models: {available_models}")
    print(f"  Missing models: {total_models - available_models}")
    
    # Check datasets
    print(f"\n📁 Dataset Status:")
    datasets = {
        "Ball Dataset": "data/combined_dataset/dataset.yaml",
        "Table Dataset": "table detector/data.yaml",
        "Pocket Dataset": "pocket detection/data.yaml"
    }
    
    for name, path in datasets.items():
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} - NOT FOUND")

if __name__ == "__main__":
    check_model_status() 
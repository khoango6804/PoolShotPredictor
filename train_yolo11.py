#!/usr/bin/env python3
"""
Train YOLOv11 on new billiards dataset
"""

import os
import yaml
from ultralytics import YOLO
import argparse

def train_yolo11_model():
    """Train YOLOv11 model on new dataset"""
    
    print("🚀 YOLOv11 Training on New Billiards Dataset")
    print("=" * 50)
    
    # Dataset info
    dataset_path = "new dataset/data.yaml"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Load dataset config
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"📊 Dataset Info:")
    print(f"  Classes: {dataset_config['nc']}")
    print(f"  Names: {dataset_config['names']}")
    print(f"  Train: {dataset_config['train']}")
    print(f"  Val: {dataset_config['val']}")
    print(f"  Test: {dataset_config['test']}")
    
    # Initialize YOLOv11 model
    print(f"\n🤖 Initializing YOLOv11 model...")
    model = YOLO('yolo11n.pt')  # Start with YOLOv11 nano
    
    # Training parameters
    training_params = {
        'data': dataset_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',
        'workers': 8,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'nbs': 64,
        'val': True,
        'plots': True,
        'project': 'runs/detect',
        'name': 'yolo11_billiards_v1',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
    }
    
    print(f"\n🎯 Starting YOLOv11 training...")
    print(f"  Epochs: {training_params['epochs']}")
    print(f"  Image size: {training_params['imgsz']}")
    print(f"  Batch size: {training_params['batch']}")
    print(f"  Project: {training_params['project']}")
    print(f"  Name: {training_params['name']}")
    
    # Start training
    try:
        results = model.train(**training_params)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: runs/detect/{training_params['name']}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📊 Final Metrics:")
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on billiards dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLOv11 model size")
    
    args = parser.parse_args()
    
    # Update training parameters
    if args.epochs != 100:
        print(f"📝 Using {args.epochs} epochs")
    if args.batch != 16:
        print(f"📝 Using batch size {args.batch}")
    if args.imgsz != 640:
        print(f"📝 Using image size {args.imgsz}")
    if args.model != "yolo11n.pt":
        print(f"📝 Using model {args.model}")
    
    # Start training
    success = train_yolo11_model()
    
    if success:
        print(f"\n🎉 YOLOv11 training completed!")
        print(f"📁 Check results in: runs/detect/yolo11_billiards_v1/")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main() 
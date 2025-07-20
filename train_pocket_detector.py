#!/usr/bin/env python3
"""
Train YOLOv11 Pocket Detector from checkpoint
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_pocket_detector():
    """Train pocket detector from checkpoint"""
    
    # Configuration
    model_name = "yolov8n.pt"  # Use YOLOv8n as base model
    dataset_path = "pocket detection/data.yaml"
    project_name = "pocket_detector"
    epochs = 100
    imgsz = 640
    batch_size = 16
    
    print("🎯 Training YOLOv11 Pocket Detector")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Check if base model exists
    if not Path(model_name).exists():
        print(f"❌ Base model not found: {model_name}")
        print("Downloading YOLOv11n from Ultralytics...")
        try:
            model = YOLO("yolov11n.pt")  # Download from Ultralytics
        except Exception as e:
            print(f"❌ Failed to download YOLOv11n: {e}")
            print("Trying with YOLOv8n instead...")
            model = YOLO("yolov8n.pt")  # Fallback to YOLOv8n
    else:
        model = YOLO(model_name)
    
    print(f"📊 Dataset: {dataset_path}")
    print(f"🎯 Classes: 10 (pocket types)")
    print(f"   - BottomLeft, BottomRight")
    print(f"   - IntersectionLeft, IntersectionRight")
    print(f"   - MediumLeft, MediumRight")
    print(f"   - SemicircleLeft, SemicircleRight")
    print(f"   - TopLeft, TopRight")
    print(f"⏱️ Epochs: {epochs}")
    print(f"🖼️ Image size: {imgsz}")
    print(f"📦 Batch size: {batch_size}")
    print("=" * 50)
    
    try:
        # Train the model
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project_name,
            name="yolo11_pocket_detector",
            patience=20,
            save=True,
            save_period=10,
            device=0,  # GPU
            workers=4,
            verbose=True
        )
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {project_name}/yolo11_pocket_detector/")
        
        # Save best model
        best_model_path = f"{project_name}/yolo11_pocket_detector/weights/best.pt"
        if Path(best_model_path).exists():
            print(f"🏆 Best model: {best_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def train_from_checkpoint():
    """Train from existing checkpoint"""
    
    checkpoint_path = "pocket_detector/yolo11_pocket_detector/weights/best.pt"
    dataset_path = "pocket detection/data.yaml"
    project_name = "pocket_detector"
    epochs = 50  # Continue training for 50 more epochs
    imgsz = 640
    batch_size = 16
    
    print("🔄 Continuing Pocket Detector Training from Checkpoint")
    print("=" * 50)
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Starting fresh training...")
        return train_pocket_detector()
    
    print(f"📁 Loading checkpoint: {checkpoint_path}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"⏱️ Additional epochs: {epochs}")
    print("=" * 50)
    
    try:
        # Load model from checkpoint
        model = YOLO(checkpoint_path)
        
        # Continue training
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project_name,
            name="yolo11_pocket_detector_continued",
            patience=20,
            save=True,
            save_period=10,
            device=0,  # GPU
            workers=4,
            verbose=True
        )
        
        print("✅ Continued training completed successfully!")
        print(f"📁 Results saved to: {project_name}/yolo11_pocket_detector_continued/")
        
        return True
        
    except Exception as e:
        print(f"❌ Continued training failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 YOLOv11 Pocket Detector Training")
    print("=" * 40)
    
    # Try to continue from checkpoint first
    success = train_from_checkpoint()
    
    if not success:
        print("🔄 Starting fresh training...")
        success = train_pocket_detector()
    
    if success:
        print("\n🎉 Pocket detector training completed!")
        print("📁 Check the results in pocket_detector/ directory")
    else:
        print("\n❌ Training failed. Please check the error messages above.") 
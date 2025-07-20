#!/usr/bin/env python3
"""
Test YOLOv11 model on real image uploaded by user
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

def test_real_image(model_path, image_path, conf=0.01):
    """Test YOLOv11 on real image"""
    
    print(f"🎯 Testing YOLOv11 on real image...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Confidence: {conf}")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    # Get image info
    height, width = image.shape[:2]
    print(f"📐 Image size: {width}x{height}")
    
    # Run detection
    results = model(image, conf=float(conf), verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            detections = len(boxes)
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            
            print(f"🎯 Total detections: {detections}")
            print(f"📊 Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
            print(f"🏷️ Classes detected: {np.unique(class_ids.astype(int))}")
            
            # Show all detections sorted by confidence
            if detections > 0:
                print(f"\n📋 All detections (sorted by confidence):")
                sorted_indices = np.argsort(confidences)[::-1]  # High to low
                
                for i, idx in enumerate(sorted_indices):
                    class_id = int(class_ids[idx])
                    conf = confidences[idx]
                    print(f"  {i+1:2d}. Class {class_id:2d}: {conf:.3f}")
                    
                    if i >= 19:  # Show first 20
                        print(f"  ... and {detections - 20} more detections")
                        break
            
            # Class distribution
            unique_classes, counts = np.unique(class_ids.astype(int), return_counts=True)
            print(f"\n📊 Class distribution:")
            for class_id, count in zip(unique_classes, counts):
                print(f"  Class {class_id:2d}: {count:3d} detections")
            
        else:
            print(f"❌ No detections")
    else:
        print(f"❌ No results")
    
    # Save result image
    if results and len(results) > 0:
        output_path = f"real_image_result.jpg"
        result_image = results[0].plot()
        cv2.imwrite(output_path, result_image)
        print(f"\n💾 Result saved: {output_path}")
    
    return {
        'detections': detections if results and len(results) > 0 and results[0].boxes is not None else 0,
        'classes': np.unique(class_ids.astype(int)) if results and len(results) > 0 and results[0].boxes is not None else []
    }

def test_multiple_confidences_real(model_path, image_path):
    """Test with multiple confidence levels"""
    
    print(f"🔍 Testing multiple confidence levels on real image...")
    print("=" * 60)
    
    confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    results_summary = {}
    
    for conf in confidence_levels:
        print(f"\n{'='*50}")
        print(f"Testing with confidence: {conf}")
        result = test_real_image(model_path, image_path, conf)
        results_summary[conf] = result
    
    # Summary
    print(f"\n📊 COMPREHENSIVE SUMMARY:")
    print(f"{'Confidence':<12} {'Detections':<12} {'Classes':<30}")
    print("-" * 60)
    for conf, result in results_summary.items():
        detections = result.get('detections', 0)
        classes = result.get('classes', [])
        class_str = ', '.join(map(str, classes)) if len(classes) > 0 else 'None'
        print(f"{conf:<12} {detections:<12} {class_str:<30}")

def main():
    print("🎯 YOLOv11 Real Image Testing")
    print("=" * 40)
    
    model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    
    # Check for common image files
    possible_images = [
        "billiards.jpg", "billiards.jpeg", "billiards.png",
        "pool.jpg", "pool.jpeg", "pool.png",
        "8ball.jpg", "8ball.jpeg", "8ball.png",
        "test.jpg", "test.jpeg", "test.png",
        "image.jpg", "image.jpeg", "image.png"
    ]
    
    image_path = None
    for img in possible_images:
        if Path(img).exists():
            image_path = img
            break
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if image_path is None:
        print(f"❌ No image found!")
        print(f"Please place your image in the current directory with one of these names:")
        for img in possible_images:
            print(f"  - {img}")
        print(f"\nOr rename your image to one of the above names.")
        return
    
    print(f"📸 Found image: {image_path}")
    
    # Test with multiple confidence levels
    test_multiple_confidences_real(model_path, image_path)
    
    print(f"\n🎉 Real image testing completed!")

if __name__ == "__main__":
    main() 
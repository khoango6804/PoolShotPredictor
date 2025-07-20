#!/usr/bin/env python3
"""
Test YOLOv11 model with low confidence
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_low_confidence(model_path, image_path, conf=0.01):
    """Test with very low confidence to see all detections"""
    
    print(f"🔍 Testing YOLOv11 with low confidence...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Confidence: {conf}")
    print("=" * 50)
    
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
    
    # Run detection with low confidence
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
                    
                    if i >= 49:  # Show first 50
                        print(f"  ... and {detections - 50} more detections")
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
        output_path = f"low_conf_test_{os.path.basename(image_path)}"
        result_image = results[0].plot()
        cv2.imwrite(output_path, result_image)
        print(f"\n💾 Result saved: {output_path}")
    
    return {
        'detections': detections if results and len(results) > 0 and results[0].boxes is not None else 0,
        'classes': np.unique(class_ids.astype(int)) if results and len(results) > 0 and results[0].boxes is not None else []
    }

def main():
    print("🔍 YOLOv11 Low Confidence Testing")
    print("=" * 40)
    
    model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    image_path = "demo_result.jpg"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Test with different confidence levels
    confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    for conf in confidence_levels:
        print(f"\n{'='*60}")
        test_low_confidence(model_path, image_path, conf)
    
    print(f"\n🎉 Low confidence testing completed!")

if __name__ == "__main__":
    import os
    main() 
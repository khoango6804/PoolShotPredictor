#!/usr/bin/env python3
"""
Test YOLOv11 model on user's billiards image
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_user_image(model_path, image_path, conf=0.01):
    """Test YOLOv11 on user's billiards image"""
    
    print(f"🎯 Testing YOLOv11 on user's billiards image...")
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
            
            # Expected vs detected
            expected_balls = 15  # 1 cue + 14 numbered balls
            detection_rate = (detections / expected_balls) * 100
            print(f"\n📈 Detection Analysis:")
            print(f"  Expected balls: {expected_balls}")
            print(f"  Detected balls: {detections}")
            print(f"  Detection rate: {detection_rate:.1f}%")
            
            # Performance assessment
            if detection_rate >= 90:
                print(f"  🏆 Performance: EXCELLENT!")
            elif detection_rate >= 80:
                print(f"  🥇 Performance: VERY GOOD!")
            elif detection_rate >= 70:
                print(f"  🥈 Performance: GOOD!")
            elif detection_rate >= 60:
                print(f"  🥉 Performance: FAIR!")
            else:
                print(f"  ⚠️ Performance: NEEDS IMPROVEMENT!")
            
        else:
            print(f"❌ No detections")
    else:
        print(f"❌ No results")
    
    # Save result image
    if results and len(results) > 0:
        output_path = f"user_billiards_result.jpg"
        result_image = results[0].plot()
        cv2.imwrite(output_path, result_image)
        print(f"\n💾 Result saved: {output_path}")
    
    return {
        'detections': detections if results and len(results) > 0 and results[0].boxes is not None else 0,
        'classes': np.unique(class_ids.astype(int)) if results and len(results) > 0 and results[0].boxes is not None else [],
        'detection_rate': detection_rate if results and len(results) > 0 and results[0].boxes is not None else 0
    }

def test_multiple_confidences_user(model_path, image_path):
    """Test with multiple confidence levels"""
    
    print(f"🔍 Testing multiple confidence levels on user's image...")
    print("=" * 60)
    
    confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    results_summary = {}
    
    for conf in confidence_levels:
        print(f"\n{'='*50}")
        print(f"Testing with confidence: {conf}")
        result = test_user_image(model_path, image_path, conf)
        results_summary[conf] = result
    
    # Summary
    print(f"\n📊 COMPREHENSIVE SUMMARY:")
    print(f"{'Confidence':<12} {'Detections':<12} {'Detection Rate':<15} {'Performance':<15} {'Classes':<30}")
    print("-" * 90)
    for conf, result in results_summary.items():
        detections = result.get('detections', 0)
        detection_rate = result.get('detection_rate', 0)
        classes = result.get('classes', [])
        class_str = ', '.join(map(str, classes)) if len(classes) > 0 else 'None'
        
        # Performance rating
        if detection_rate >= 90:
            performance = "EXCELLENT"
        elif detection_rate >= 80:
            performance = "VERY GOOD"
        elif detection_rate >= 70:
            performance = "GOOD"
        elif detection_rate >= 60:
            performance = "FAIR"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        print(f"{conf:<12} {detections:<12} {detection_rate:<15.1f}% {performance:<15} {class_str:<30}")

def main():
    print("🎯 YOLOv11 User Billiards Testing")
    print("=" * 50)
    
    model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    image_path = "user_billiards_image.jpg"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print(f"Please run create_user_billiards_image.py first")
        return
    
    # Test with multiple confidence levels
    test_multiple_confidences_user(model_path, image_path)
    
    print(f"\n🎉 User billiards testing completed!")

if __name__ == "__main__":
    main() 
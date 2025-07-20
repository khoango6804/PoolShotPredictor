#!/usr/bin/env python3
"""
Quick Test Multi-Ball Detection
Script test nhanh để kiểm tra việc detect nhiều bóng
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path

def test_multi_ball_detection(model_path, image_path):
    """Test detect nhiều bóng trên một hình ảnh"""
    
    print(f"🎯 Testing multi-ball detection...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Không thể load image: {image_path}")
        return
    
    print(f"Image size: {frame.shape}")
    
    # Test different confidence thresholds
    confidence_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    iou_thresholds = [0.1, 0.2, 0.3]
    
    best_result = {
        'confidence': 0.3,
        'iou': 0.3,
        'ball_count': 0,
        'total_detections': 0,
        'avg_confidence': 0
    }
    
    print(f"\n🔍 Testing different parameters...")
    
    for conf in confidence_thresholds:
        for iou in iou_thresholds:
            print(f"\n📊 Testing: conf={conf}, iou={iou}")
            
            # Run detection
            results = model(frame, conf=conf, iou=iou, verbose=False)
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Get detections
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    
                    # Filter ball detections (class 0)
                    ball_indices = np.where(classes == 0)[0]
                    ball_detections = len(ball_indices)
                    total_detections = len(classes)
                    avg_confidence = np.mean(confidences[ball_indices]) if len(ball_indices) > 0 else 0
                    
                    print(f"  🎱 Balls detected: {ball_detections}")
                    print(f"  📊 Total detections: {total_detections}")
                    print(f"  📈 Avg ball confidence: {avg_confidence:.3f}")
                    
                    # Save best result
                    if ball_detections > best_result['ball_count']:
                        best_result = {
                            'confidence': conf,
                            'iou': iou,
                            'ball_count': ball_detections,
                            'total_detections': total_detections,
                            'avg_confidence': avg_confidence
                        }
                        
                        # Save annotated image
                        annotated_frame = results[0].plot()
                        cv2.imwrite("best_multi_ball_detection.jpg", annotated_frame)
                        print(f"  💾 Saved best result: best_multi_ball_detection.jpg")
                else:
                    print(f"  ❌ No detections")
            else:
                print(f"  ❌ No results")
    
    print(f"\n🏆 BEST RESULT:")
    print(f"  Confidence: {best_result['confidence']}")
    print(f"  IOU: {best_result['iou']}")
    print(f"  Balls detected: {best_result['ball_count']}")
    print(f"  Total detections: {best_result['total_detections']}")
    print(f"  Avg confidence: {best_result['avg_confidence']:.3f}")
    
    return best_result

def test_with_optimized_settings(model_path, image_path):
    """Test với cấu hình tối ưu cho multi-ball"""
    
    print(f"\n🚀 Testing with optimized settings for multi-ball detection...")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    frame = cv2.imread(str(image_path))
    
    # Optimized settings for multi-ball detection
    optimized_config = {
        'confidence': 0.15,  # Lower confidence threshold
        'iou': 0.2,         # Lower IOU threshold
        'max_det': 100      # Higher max detections
    }
    
    print(f"Optimized config: {optimized_config}")
    
    # Run detection with optimized settings
    results = model(frame, 
                   conf=optimized_config['confidence'],
                   iou=optimized_config['iou'],
                   verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Count ball detections
            ball_indices = np.where(classes == 0)[0]
            ball_detections = len(ball_indices)
            
            print(f"✅ Optimized detection results:")
            print(f"  🎱 Total balls detected: {ball_detections}")
            print(f"  🎯 Ball classes: {classes[ball_indices]}")
            print(f"  📈 Ball confidences: {confidences[ball_indices]}")
            
            # Save result
            annotated_frame = results[0].plot()
            cv2.imwrite("optimized_multi_ball_result.jpg", annotated_frame)
            print(f"  💾 Saved result: optimized_multi_ball_result.jpg")
            
            return ball_detections
        else:
            print("❌ No detections found")
            return 0
    else:
        print("❌ No results")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Quick test multi-ball detection")
    parser.add_argument("--model", default="models/billiards_model.pt", help="Path to YOLO model")
    parser.add_argument("--image", required=True, help="Path to test image")
    
    args = parser.parse_args()
    
    # Test with different parameters
    best_result = test_multi_ball_detection(args.model, args.image)
    
    # Test with optimized settings
    optimized_result = test_with_optimized_settings(args.model, args.image)
    
    print(f"\n📊 COMPARISON:")
    print(f"  Default best: {best_result['ball_count']} balls")
    print(f"  Optimized: {optimized_result} balls")
    
    if optimized_result > best_result['ball_count']:
        print(f"✅ Optimized settings detected more balls!")
    else:
        print(f"⚠️ Default settings performed better")

if __name__ == "__main__":
    main() 
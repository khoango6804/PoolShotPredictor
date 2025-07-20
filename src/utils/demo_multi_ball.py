#!/usr/bin/env python3
"""
Demo Multi-Ball Detection
Script demo đơn giản để test detect nhiều bóng
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path

def demo_multi_ball_detection(model_path, image_path):
    """Demo detect nhiều bóng"""
    
    print(f"🎯 Demo Multi-Ball Detection")
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
    
    # Test với cấu hình tối ưu cho multi-ball
    print(f"\n🚀 Testing with optimized multi-ball settings...")
    
    # Cấu hình tối ưu cho detect nhiều bóng
    optimized_settings = {
        'confidence': 0.15,  # Confidence threshold thấp hơn
        'iou': 0.2,         # IOU threshold thấp hơn
        'max_det': 100      # Số lượng detection cao hơn
    }
    
    print(f"Optimized settings: {optimized_settings}")
    
    # Run detection
    results = model(frame, 
                   conf=optimized_settings['confidence'],
                   iou=optimized_settings['iou'],
                   verbose=False)
    
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
            
            print(f"\n✅ Detection Results:")
            print(f"  🎱 Balls detected: {ball_detections}")
            print(f"  📊 Total detections: {total_detections}")
            
            if ball_detections > 0:
                print(f"  🎯 Ball confidences: {confidences[ball_indices]}")
                print(f"  📈 Average ball confidence: {np.mean(confidences[ball_indices]):.3f}")
            
            # Save annotated image
            annotated_frame = results[0].plot()
            output_file = "multi_ball_detection_result.jpg"
            cv2.imwrite(output_file, annotated_frame)
            print(f"  💾 Saved result: {output_file}")
            
            # Draw additional statistics on image
            stats_frame = annotated_frame.copy()
            cv2.putText(stats_frame, f"Balls: {ball_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(stats_frame, f"Total: {total_detections}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(stats_frame, f"Conf: {optimized_settings['confidence']}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite("multi_ball_detection_with_stats.jpg", stats_frame)
            print(f"  💾 Saved with stats: multi_ball_detection_with_stats.jpg")
            
            return ball_detections
        else:
            print("❌ No detections found")
            return 0
    else:
        print("❌ No results")
        return 0

def compare_with_default(model_path, image_path):
    """So sánh với cấu hình mặc định"""
    
    print(f"\n🔍 Comparing with default settings...")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    frame = cv2.imread(str(image_path))
    
    # Default settings
    default_settings = {
        'confidence': 0.3,
        'iou': 0.3,
        'max_det': 30
    }
    
    print(f"Default settings: {default_settings}")
    
    # Run detection with default settings
    results = model(frame, 
                   conf=default_settings['confidence'],
                   iou=default_settings['iou'],
                   verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            ball_indices = np.where(classes == 0)[0]
            ball_detections = len(ball_indices)
            
            print(f"Default detection: {ball_detections} balls")
            
            # Save default result
            annotated_frame = results[0].plot()
            cv2.imwrite("default_detection_result.jpg", annotated_frame)
            print(f"Saved default result: default_detection_result.jpg")
            
            return ball_detections
        else:
            print("Default: No detections found")
            return 0
    else:
        print("Default: No results")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Demo multi-ball detection")
    parser.add_argument("--model", default="models/billiards_model.pt", help="Path to YOLO model")
    parser.add_argument("--image", required=True, help="Path to test image")
    
    args = parser.parse_args()
    
    # Test optimized settings
    optimized_result = demo_multi_ball_detection(args.model, args.image)
    
    # Compare with default settings
    default_result = compare_with_default(args.model, args.image)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  Default settings: {default_result} balls")
    print(f"  Optimized settings: {optimized_result} balls")
    
    if optimized_result > default_result:
        print(f"✅ Optimized settings detected {optimized_result - default_result} more balls!")
    elif optimized_result == default_result:
        print(f"⚠️ Both settings detected the same number of balls")
    else:
        print(f"⚠️ Default settings detected {default_result - optimized_result} more balls")

if __name__ == "__main__":
    main() 
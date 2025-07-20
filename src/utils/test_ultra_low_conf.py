#!/usr/bin/env python3
"""
Test Ultra Low Confidence Detection
Test với confidence rất thấp để xem có detect được nhiều bóng hơn không
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse

def test_ultra_low_confidence(model_path, image_path):
    """Test với confidence rất thấp"""
    
    print(f"🎯 Testing with ultra low confidence...")
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
    
    # Test với confidence rất thấp
    ultra_low_conf = 0.01  # 1% confidence
    print(f"\n🔍 Testing with confidence: {ultra_low_conf}")
    
    # Run detection
    results = model(frame, conf=ultra_low_conf, iou=0.1, verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Get detections
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            print(f"\n✅ Ultra Low Confidence Results:")
            print(f"  📊 Total detections: {len(classes)}")
            print(f"  🎯 All classes: {classes}")
            print(f"  📈 All confidences: {confidences}")
            
            # Filter ball detections (class 0)
            ball_indices = np.where(classes == 0)[0]
            ball_detections = len(ball_indices)
            
            print(f"  🎱 Balls detected: {ball_detections}")
            if ball_detections > 0:
                print(f"  🎯 Ball confidences: {confidences[ball_indices]}")
                print(f"  📈 Average ball confidence: {np.mean(confidences[ball_indices]):.3f}")
            
            # Save result
            annotated_frame = results[0].plot()
            cv2.imwrite("ultra_low_conf_detection.jpg", annotated_frame)
            print(f"  💾 Saved: ultra_low_conf_detection.jpg")
            
            return ball_detections
        else:
            print("❌ No detections found")
            return 0
    else:
        print("❌ No results")
        return 0

def test_different_models(image_path):
    """Test với các model khác nhau"""
    
    models_to_test = [
        "models/billiards_model.pt",
        "runs/ball_classification/yolov8m_correct/weights/best.pt"
    ]
    
    results = {}
    
    for model_path in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_path}")
        print(f"{'='*50}")
        
        try:
            ball_count = test_ultra_low_confidence(model_path, image_path)
            results[model_path] = ball_count
        except Exception as e:
            print(f"❌ Error with model {model_path}: {e}")
            results[model_path] = 0
    
    # Summary
    print(f"\n📊 SUMMARY:")
    for model_path, ball_count in results.items():
        print(f"  {model_path}: {ball_count} balls")

def main():
    parser = argparse.ArgumentParser(description="Test ultra low confidence detection")
    parser.add_argument("--image", required=True, help="Path to test image")
    
    args = parser.parse_args()
    
    test_different_models(args.image)

if __name__ == "__main__":
    main() 
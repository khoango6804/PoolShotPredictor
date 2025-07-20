#!/usr/bin/env python3
"""
Optimize Multi-Ball Detection
Tối ưu hóa các tham số để detect được nhiều bóng cùng lúc
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import time

def test_detection_parameters(model_path, image_path, output_dir="optimization_results"):
    """
    Test các tham số khác nhau để tìm cấu hình tốt nhất cho multi-ball detection
    """
    
    # Load model
    print(f"🎯 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Load test image
    print(f"🖼️ Loading image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Không thể load image: {image_path}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Test parameters
    confidence_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_detections = [50, 100, 150, 200]
    
    best_config = {
        'confidence': 0.3,
        'iou': 0.3,
        'max_detections': 30,
        'total_balls': 0,
        'unique_balls': 0,
        'avg_confidence': 0
    }
    
    print("\n🔍 Testing different configurations...")
    
    for conf in confidence_thresholds:
        for iou in iou_thresholds:
            for max_det in max_detections:
                print(f"\n📊 Testing: conf={conf}, iou={iou}, max_det={max_det}")
                
                # Run detection
                start_time = time.time()
                results = model(frame, conf=conf, iou=iou, verbose=False)
                inference_time = time.time() - start_time
                
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
                        unique_balls = len(set(classes[ball_indices])) if len(ball_indices) > 0 else 0
                        avg_confidence = np.mean(confidences[ball_indices]) if len(ball_indices) > 0 else 0
                        
                        print(f"  🎱 Balls detected: {ball_detections}")
                        print(f"  🎯 Unique balls: {unique_balls}")
                        print(f"  📈 Avg confidence: {avg_confidence:.3f}")
                        print(f"  ⏱️ Inference time: {inference_time:.3f}s")
                        
                        # Save best result
                        if ball_detections > best_config['total_balls']:
                            best_config = {
                                'confidence': conf,
                                'iou': iou,
                                'max_detections': max_det,
                                'total_balls': ball_detections,
                                'unique_balls': unique_balls,
                                'avg_confidence': avg_confidence,
                                'inference_time': inference_time
                            }
                            
                            # Save annotated image
                            annotated_frame = results[0].plot()
                            output_file = output_path / f"best_config_conf{conf}_iou{iou}_max{max_det}.jpg"
                            cv2.imwrite(str(output_file), annotated_frame)
                            print(f"  💾 Saved: {output_file}")
                    else:
                        print(f"  ❌ No detections")
                else:
                    print(f"  ❌ No results")
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"  Confidence: {best_config['confidence']}")
    print(f"  IOU: {best_config['iou']}")
    print(f"  Max Detections: {best_config['max_detections']}")
    print(f"  Total Balls: {best_config['total_balls']}")
    print(f"  Unique Balls: {best_config['unique_balls']}")
    print(f"  Avg Confidence: {best_config['avg_confidence']:.3f}")
    print(f"  Inference Time: {best_config['inference_time']:.3f}s")
    
    # Generate optimized config
    generate_optimized_config(best_config, output_path)
    
    return best_config

def generate_optimized_config(best_config, output_path):
    """Generate optimized configuration file"""
    
    config_content = f'''# Optimized Configuration for Multi-Ball Detection
# Generated from optimization test

# Model configurations
YOLO_MODEL = "yolov8m.pt"
CONFIDENCE_THRESHOLD = {best_config['confidence']}  # Optimized for multi-ball detection
IOU_THRESHOLD = {best_config['iou']}  # Optimized to allow overlapping balls
MAX_DETECTIONS = {best_config['max_detections']}  # Increased for more balls

# Size filters - Relaxed for better detection
MIN_BALL_SIZE = 15  # Reduced from 20
MAX_BALL_SIZE = 150  # Increased from 100

# Additional optimizations
ENABLE_SIZE_FILTERING = False  # Disable size filtering for balls
ENABLE_OVERLAP_FILTERING = False  # Disable overlap filtering for balls

# Performance settings
BATCH_SIZE = 1
DEVICE = "auto"  # Use GPU if available

# Results from optimization:
# Total Balls Detected: {best_config['total_balls']}
# Unique Balls: {best_config['unique_balls']}
# Average Confidence: {best_config['avg_confidence']:.3f}
# Inference Time: {best_config['inference_time']:.3f}s
'''
    
    config_file = output_path / "optimized_config.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n💾 Generated optimized config: {config_file}")

def test_with_optimized_settings(model_path, image_path, optimized_config):
    """Test với cấu hình đã tối ưu"""
    
    print(f"\n🚀 Testing with optimized settings...")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    frame = cv2.imread(str(image_path))
    
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
            print(f"  📈 Confidences: {confidences[ball_indices]}")
            
            # Save result
            annotated_frame = results[0].plot()
            cv2.imwrite("optimized_detection_result.jpg", annotated_frame)
            print(f"  💾 Saved result: optimized_detection_result.jpg")
            
            return ball_detections
        else:
            print("❌ No detections found")
            return 0
    else:
        print("❌ No results")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Optimize multi-ball detection parameters")
    parser.add_argument("--model", default="models/billiards_model.pt", help="Path to YOLO model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--output", default="optimization_results", help="Output directory")
    parser.add_argument("--test-only", action="store_true", help="Only test with current settings")
    
    args = parser.parse_args()
    
    if args.test_only:
        # Test với cấu hình hiện tại
        optimized_config = {
            'confidence': 0.3,
            'iou': 0.3,
            'max_detections': 30
        }
        test_with_optimized_settings(args.model, args.image, optimized_config)
    else:
        # Run full optimization
        best_config = test_detection_parameters(args.model, args.image, args.output)
        
        if best_config:
            # Test với cấu hình tốt nhất
            test_with_optimized_settings(args.model, args.image, best_config)

if __name__ == "__main__":
    main() 
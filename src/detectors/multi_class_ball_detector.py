#!/usr/bin/env python3
"""
Multi-Class Ball Detector
Detect bóng sử dụng tất cả các class 0-15 thay vì chỉ class 0
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path

# Ball class mapping for 8-ball pool
BALL_CLASS_MAPPING = {
    0: "cue_ball",      # Bi trắng
    1: "solid_1", 2: "solid_2", 3: "solid_3", 4: "solid_4", 
    5: "solid_5", 6: "solid_6", 7: "solid_7",
    8: "eight_ball",    # Bi đen số 8
    9: "stripe_9", 10: "stripe_10", 11: "stripe_11", 12: "stripe_12",
    13: "stripe_13", 14: "stripe_14", 15: "stripe_15",
    94: "unknown_ball"  # Class đặc biệt
}

# Colors for different ball types
BALL_COLORS = {
    0: (255, 255, 255),  # White for cue ball
    1: (255, 0, 0),      # Red for solid 1
    2: (0, 255, 0),      # Green for solid 2
    3: (0, 0, 255),      # Blue for solid 3
    4: (255, 255, 0),    # Yellow for solid 4
    5: (255, 0, 255),    # Magenta for solid 5
    6: (0, 255, 255),    # Cyan for solid 6
    7: (128, 0, 0),      # Dark red for solid 7
    8: (0, 0, 0),        # Black for 8-ball
    9: (255, 128, 0),    # Orange for stripe 9
    10: (128, 255, 0),   # Light green for stripe 10
    11: (0, 128, 255),   # Light blue for stripe 11
    12: (255, 0, 128),   # Pink for stripe 12
    13: (128, 0, 255),   # Purple for stripe 13
    14: (0, 255, 128),   # Light cyan for stripe 14
    15: (255, 128, 128), # Light red for stripe 15
    94: (128, 128, 128)  # Gray for unknown
}

def is_ball_class(class_id):
    """Check if class_id is a ball class (0-15, 94)"""
    return class_id in BALL_CLASS_MAPPING

def detect_all_balls(model_path, image_path, confidence=0.15, iou=0.2):
    """Detect tất cả các loại bóng sử dụng tất cả các class"""
    
    print(f"🎯 Multi-Class Ball Detection")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Confidence: {confidence}, IOU: {iou}")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Không thể load image: {image_path}")
        return None, None
    
    print(f"Image size: {frame.shape}")
    
    # Run detection
    results = model(frame, conf=confidence, iou=iou, verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Get detections
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Filter ball detections (all ball classes)
            ball_indices = [i for i, cls in enumerate(classes) if is_ball_class(int(cls))]
            ball_detections = len(ball_indices)
            
            print(f"\n✅ Multi-Class Detection Results:")
            print(f"  🎱 Total balls detected: {ball_detections}")
            print(f"  📊 Total detections: {len(classes)}")
            
            if ball_detections > 0:
                ball_classes = classes[ball_indices]
                ball_confidences = confidences[ball_indices]
                ball_boxes = xyxy[ball_indices]
                
                print(f"  🎯 Ball classes: {ball_classes}")
                print(f"  📈 Ball confidences: {ball_confidences}")
                
                # Group by ball type
                ball_types = {}
                for i, cls in enumerate(ball_classes):
                    class_id = int(cls)
                    ball_type = BALL_CLASS_MAPPING.get(class_id, f"ball_{class_id}")
                    if ball_type not in ball_types:
                        ball_types[ball_type] = []
                    ball_types[ball_type].append({
                        'confidence': ball_confidences[i],
                        'bbox': ball_boxes[i],
                        'class_id': class_id
                    })
                
                print(f"  🎱 Ball types detected:")
                for ball_type, detections in ball_types.items():
                    print(f"    {ball_type}: {len(detections)} balls")
                
                return ball_types, results[0]
            else:
                print("  ❌ No balls detected")
                return {}, results[0]
        else:
            print("❌ No detections found")
            return {}, None
    else:
        print("❌ No results")
        return {}, None

def draw_multi_class_balls(frame, ball_types, original_result=None):
    """Draw tất cả các loại bóng với màu sắc khác nhau"""
    
    annotated_frame = frame.copy()
    
    # Draw all ball detections
    for ball_type, detections in ball_types.items():
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            
            # Get color for this ball type
            color = BALL_COLORS.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1)), (int(x2), int(y2)), 
                         color, 2)
            
            # Draw ball number in center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 0), -1)
            
            # Draw label
            label = f"{ball_type}: {confidence:.2f}"
            cv2.putText(annotated_frame, label,
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # If we have original result, also draw non-ball detections
    if original_result:
        boxes = original_result.boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for i, (cls, conf, box) in enumerate(zip(classes, confidences, xyxy)):
                class_id = int(cls)
                if not is_ball_class(class_id):  # Only draw non-ball detections
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, 
                                 (int(x1), int(y1)), (int(x2), int(y2)), 
                                 (128, 128, 128), 1)  # Gray for non-balls
                    cv2.putText(annotated_frame, f"class_{class_id}: {conf:.2f}",
                               (int(x1), int(y2) + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    return annotated_frame

def compare_with_single_class(model_path, image_path):
    """So sánh với việc chỉ detect class 0"""
    
    print(f"\n🔍 Comparing with single class detection (class 0 only)...")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    frame = cv2.imread(str(image_path))
    
    # Run detection
    results = model(frame, conf=0.15, iou=0.2, verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            ball_indices = np.where(classes == 0)[0]  # Only class 0
            single_class_balls = len(ball_indices)
            
            print(f"Single class (class 0 only): {single_class_balls} balls")
            return single_class_balls
        else:
            print("Single class: No detections")
            return 0
    else:
        print("Single class: No results")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Multi-class ball detection")
    parser.add_argument("--model", default="yolov8m.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--confidence", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="IOU threshold")
    
    args = parser.parse_args()
    
    # Multi-class detection
    ball_types, original_result = detect_all_balls(args.model, args.image, args.confidence, args.iou)
    
    if ball_types:
        # Load image for drawing
        frame = cv2.imread(str(args.image))
        
        # Draw results
        annotated_frame = draw_multi_class_balls(frame, ball_types, original_result)
        
        # Save result
        output_file = "multi_class_ball_detection.jpg"
        cv2.imwrite(output_file, annotated_frame)
        print(f"\n💾 Saved multi-class result: {output_file}")
        
        # Add statistics
        stats_frame = annotated_frame.copy()
        total_balls = sum(len(detections) for detections in ball_types.values())
        cv2.putText(stats_frame, f"Total Balls: {total_balls}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stats_frame, f"Ball Types: {len(ball_types)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(stats_frame, f"Conf: {args.confidence}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite("multi_class_ball_detection_with_stats.jpg", stats_frame)
        print(f"💾 Saved with stats: multi_class_ball_detection_with_stats.jpg")
        
        # Compare with single class
        single_class_count = compare_with_single_class(args.model, args.image)
        
        # Summary
        print(f"\n📊 COMPARISON:")
        print(f"  Single class (class 0 only): {single_class_count} balls")
        print(f"  Multi-class (all ball classes): {total_balls} balls")
        
        if total_balls > single_class_count:
            improvement = total_balls - single_class_count
            print(f"✅ Multi-class detected {improvement} more balls!")
        elif total_balls == single_class_count:
            print(f"⚠️ Both methods detected the same number of balls")
        else:
            print(f"⚠️ Single class detected more balls")

if __name__ == "__main__":
    main() 
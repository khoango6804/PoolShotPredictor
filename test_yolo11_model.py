#!/usr/bin/env python3
"""
Test YOLOv11 model on videos and images
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path

# Updated class mapping for 23 classes
BALL_CLASS_MAPPING = {
    0: "bag1", 1: "bag2", 2: "bag3", 3: "bag4", 4: "bag5", 5: "bag6",
    6: "ball0", 7: "ball1", 8: "ball10", 9: "ball11", 10: "ball12", 11: "ball13",
    12: "ball14", 13: "ball15", 14: "ball2", 15: "ball3", 16: "ball4", 17: "ball5",
    18: "ball6", 19: "ball7", 20: "ball8", 21: "ball9", 22: "flag"
}

# Colors for different ball types
BALL_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
    6: (255, 255, 255), 7: (255, 128, 0), 8: (128, 255, 0), 9: (0, 128, 255), 10: (255, 0, 128), 11: (128, 0, 255),
    12: (0, 255, 128), 13: (255, 128, 128), 14: (128, 128, 0), 15: (0, 128, 128), 16: (128, 0, 128), 17: (255, 165, 0),
    18: (0, 255, 0), 19: (128, 0, 0), 20: (0, 0, 0), 21: (255, 215, 0), 22: (255, 255, 255)
}

def test_image(model_path, image_path, output_path=None, conf=0.5):
    """Test YOLOv11 model on single image"""
    
    print(f"🖼️ Testing image: {image_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    # Run detection
    results = model(image, conf=float(conf), verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Get detections
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Draw detections
            annotated_image = image.copy()
            
            for i, (cls, conf, box) in enumerate(zip(classes, confidences, xyxy)):
                class_id = int(cls)
                x1, y1, x2, y2 = box
                
                # Get color and label
                color = BALL_COLORS.get(class_id, (255, 255, 255))
                label = BALL_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # Draw bounding box
                cv2.rectangle(annotated_image, 
                             (int(x1), int(y1)), (int(x2), int(y2)), 
                             color, 2)
                
                # Draw label
                label_text = f"{label}: {conf:.2f}"
                cv2.putText(annotated_image, label_text,
                           (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save result
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                print(f"💾 Result saved to: {output_path}")
            
            print(f"✅ Detected {len(classes)} objects")
            return len(classes)
        else:
            print("❌ No detections found")
            return 0
    else:
        print("❌ No results from model")
        return 0

def test_video(model_path, video_path, output_path=None, max_frames=None, conf=0.5):
    """Test YOLOv11 model on video"""
    
    print(f"🎬 Testing video: {video_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    frame_count = 0
    total_detections = 0
    detection_counts = []
    
    print(f"🎯 Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Limit frames for testing
        if max_frames and frame_count > max_frames:
            break
        
        # Progress indicator
        if frame_count % 50 == 0:
            progress = (frame_count / min(total_frames, max_frames if max_frames else total_frames)) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_count}")
        
        # Run detection
        results = model(frame, conf=float(conf), verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Get detections
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                
                # Draw detections
                annotated_frame = frame.copy()
                
                for i, (cls, conf, box) in enumerate(zip(classes, confidences, xyxy)):
                    class_id = int(cls)
                    x1, y1, x2, y2 = box
                    
                    # Get color and label
                    color = BALL_COLORS.get(class_id, (255, 255, 255))
                    label = BALL_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, 
                                 (int(x1), int(y1)), (int(x2), int(y2)), 
                                 color, 2)
                    
                    # Draw label
                    label_text = f"{label}: {conf:.2f}"
                    cv2.putText(annotated_frame, label_text,
                               (int(x1), int(y1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Add frame info
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Detections: {len(classes)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update statistics
                total_detections += len(classes)
                detection_counts.append(len(classes))
                
                # Write frame
                if output_path:
                    out.write(annotated_frame)
            else:
                detection_counts.append(0)
                if output_path:
                    out.write(frame)
        else:
            detection_counts.append(0)
            if output_path:
                out.write(frame)
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    
    # Print statistics
    avg_detections = np.mean(detection_counts) if detection_counts else 0
    max_detections = max(detection_counts) if detection_counts else 0
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  🎬 Frames processed: {frame_count}")
    print(f"  🎯 Total detections: {total_detections}")
    print(f"  📈 Average detections per frame: {avg_detections:.1f}")
    print(f"  🏆 Maximum detections in single frame: {max_detections}")
    if output_path:
        print(f"  💾 Output video: {output_path}")
    
    return {
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'avg_detections_per_frame': avg_detections,
        'max_detections_per_frame': max_detections
    }

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv11 model")
    parser.add_argument("--model", default="runs/detect/yolo11_billiards_v1/weights/best.pt", 
                       help="Path to YOLOv11 model")
    parser.add_argument("--input", required=True, help="Input image or video path")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--mode", choices=['image', 'video'], default='video', help="Test mode")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (video mode)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Make sure training has completed and model exists")
        return
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"❌ Input not found: {args.input}")
        return
    
    print(f"🤖 YOLOv11 Model Testing")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Confidence: {args.conf}")
    print("=" * 50)
    
    if args.mode == 'image':
        test_image(args.model, args.input, args.output, args.conf)
    else:
        test_video(args.model, args.input, args.output, args.max_frames, args.conf)
    
    print(f"\n✅ Testing completed!")

if __name__ == "__main__":
    main() 
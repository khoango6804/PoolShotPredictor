#!/usr/bin/env python3
"""
Multi-Class Video Ball Detector
Detect bóng trên video sử dụng tất cả các class 0-15
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import time

# Ball class mapping for new dataset (23 classes)
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

def is_ball_class(class_id):
    """Check if class_id is a ball class (0-15)"""
    return class_id in BALL_CLASS_MAPPING

def process_video(model_path, video_path, output_path="output_video.mp4", max_frames=None):
    """Xử lý video với multi-class ball detection"""
    
    print(f"🎬 Multi-Class Video Ball Detection")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không thể open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    frame_count = 0
    total_balls_detected = 0
    ball_counts_per_frame = []
    inference_times = []
    ball_types_detected = set()
    
    print(f"\n🎯 Processing video...")
    
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
        start_time = time.time()
        results = model(frame, conf=0.5, iou=0.2, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
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
                
                # Update statistics
                total_balls_detected += ball_detections
                ball_counts_per_frame.append(ball_detections)
                
                # Group by ball type for this frame
                ball_types = {}
                for i in ball_indices:
                    class_id = int(classes[i])
                    ball_type = BALL_CLASS_MAPPING.get(class_id, f"ball_{class_id}")
                    ball_types_detected.add(ball_type)
                    
                    if ball_type not in ball_types:
                        ball_types[ball_type] = []
                    ball_types[ball_type].append({
                        'confidence': confidences[i],
                        'bbox': xyxy[i],
                        'class_id': class_id
                    })
                
                # Draw detections
                annotated_frame = draw_multi_class_balls(frame, ball_types, results[0])
                
                # Add statistics text
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Balls: {ball_detections}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                fps_text = f"FPS: {1/inference_time:.1f}" if inference_time > 0 else "FPS: N/A"
                cv2.putText(annotated_frame, fps_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Total: {total_balls_detected}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame
                out.write(annotated_frame)
            else:
                # No detections, write original frame with stats
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Balls: 0", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out.write(frame)
                ball_counts_per_frame.append(0)
        else:
            # No results, write original frame
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Balls: 0", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame)
            ball_counts_per_frame.append(0)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print final statistics
    avg_balls = np.mean(ball_counts_per_frame) if ball_counts_per_frame else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    max_balls = max(ball_counts_per_frame) if ball_counts_per_frame else 0
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  🎬 Frames processed: {frame_count}")
    print(f"  🎱 Total balls detected: {total_balls_detected}")
    print(f"  📈 Average balls per frame: {avg_balls:.1f}")
    print(f"  🏆 Maximum balls in single frame: {max_balls}")
    print(f"  ⏱️ Average inference time: {avg_inference_time:.3f}s")
    avg_fps = f"{1/avg_inference_time:.1f}" if avg_inference_time > 0 else "N/A"
    print(f"  🎯 Average FPS: {avg_fps}")
    print(f"  🎱 Ball types detected: {len(ball_types_detected)}")
    print(f"  📋 Ball types: {sorted(ball_types_detected)}")
    print(f"  💾 Output video: {output_path}")
    
    return {
        'frames_processed': frame_count,
        'total_balls': total_balls_detected,
        'avg_balls_per_frame': avg_balls,
        'max_balls_per_frame': max_balls,
        'avg_inference_time': avg_inference_time,
        'ball_types': sorted(ball_types_detected)
    }

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
            
            # Draw label (shorter for video)
            label = f"{ball_type.split('_')[-1]}: {confidence:.2f}"
            cv2.putText(annotated_frame, label,
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # If we have original result, also draw non-ball detections (optional)
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
    
    return annotated_frame

def main():
    parser = argparse.ArgumentParser(description="Multi-class video ball detection")
    parser.add_argument("--model", default="yolov8m.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="multi_class_video_output.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    # Process video
    stats = process_video(args.model, args.video, args.output, args.max_frames)
    
    if stats:
        print(f"\n✅ Video processing completed successfully!")
        print(f"🎬 Check the output video: {args.output}")

if __name__ == "__main__":
    main() 
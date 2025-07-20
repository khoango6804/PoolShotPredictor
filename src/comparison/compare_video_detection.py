#!/usr/bin/env python3
"""
Compare Single-Class vs Multi-Class Video Detection
So sánh hiệu quả detect bóng giữa chỉ class 0 và tất cả class 0-15
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import json

def is_ball_class(class_id):
    """Check if class_id is a ball class (0-15, 94)"""
    ball_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 94}
    return class_id in ball_classes

def process_video_single_class(model_path, video_path, max_frames=None):
    """Xử lý video chỉ với class 0"""
    
    print(f"🎯 Single-Class Detection (Class 0 only)")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không thể open video: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Statistics
    frame_count = 0
    total_balls_detected = 0
    ball_counts_per_frame = []
    inference_times = []
    
    print(f"Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Limit frames for testing
        if max_frames and frame_count > max_frames:
            break
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / min(total_frames, max_frames if max_frames else total_frames)) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_count}")
        
        # Run detection
        start_time = time.time()
        results = model(frame, conf=0.15, iou=0.2, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Get detections
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                # Filter only class 0 (cue ball)
                ball_indices = np.where(classes == 0)[0]
                ball_detections = len(ball_indices)
                
                # Update statistics
                total_balls_detected += ball_detections
                ball_counts_per_frame.append(ball_detections)
            else:
                ball_counts_per_frame.append(0)
        else:
            ball_counts_per_frame.append(0)
    
    # Cleanup
    cap.release()
    
    # Calculate statistics
    avg_balls = np.mean(ball_counts_per_frame) if ball_counts_per_frame else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    max_balls = max(ball_counts_per_frame) if ball_counts_per_frame else 0
    
    stats = {
        'method': 'single_class',
        'frames_processed': frame_count,
        'total_balls': total_balls_detected,
        'avg_balls_per_frame': avg_balls,
        'max_balls_per_frame': max_balls,
        'avg_inference_time': avg_inference_time,
        'avg_fps': 1/avg_inference_time if avg_inference_time > 0 else 0
    }
    
    print(f"Single-Class Results:")
    print(f"  Frames: {frame_count}, Total balls: {total_balls_detected}")
    print(f"  Avg balls/frame: {avg_balls:.1f}, Max balls: {max_balls}")
    print(f"  Avg FPS: {stats['avg_fps']:.1f}")
    
    return stats

def process_video_multi_class(model_path, video_path, max_frames=None):
    """Xử lý video với tất cả class 0-15"""
    
    print(f"🎯 Multi-Class Detection (All ball classes)")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không thể open video: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Statistics
    frame_count = 0
    total_balls_detected = 0
    ball_counts_per_frame = []
    inference_times = []
    ball_types_detected = set()
    
    print(f"Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Limit frames for testing
        if max_frames and frame_count > max_frames:
            break
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / min(total_frames, max_frames if max_frames else total_frames)) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_count}")
        
        # Run detection
        start_time = time.time()
        results = model(frame, conf=0.15, iou=0.2, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Get detections
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                # Filter all ball classes
                ball_indices = [i for i, cls in enumerate(classes) if is_ball_class(int(cls))]
                ball_detections = len(ball_indices)
                
                # Update statistics
                total_balls_detected += ball_detections
                ball_counts_per_frame.append(ball_detections)
                
                # Track ball types
                for i in ball_indices:
                    class_id = int(classes[i])
                    ball_types_detected.add(class_id)
            else:
                ball_counts_per_frame.append(0)
        else:
            ball_counts_per_frame.append(0)
    
    # Cleanup
    cap.release()
    
    # Calculate statistics
    avg_balls = np.mean(ball_counts_per_frame) if ball_counts_per_frame else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    max_balls = max(ball_counts_per_frame) if ball_counts_per_frame else 0
    
    stats = {
        'method': 'multi_class',
        'frames_processed': frame_count,
        'total_balls': total_balls_detected,
        'avg_balls_per_frame': avg_balls,
        'max_balls_per_frame': max_balls,
        'avg_inference_time': avg_inference_time,
        'avg_fps': 1/avg_inference_time if avg_inference_time > 0 else 0,
        'ball_types_detected': sorted(list(ball_types_detected))
    }
    
    print(f"Multi-Class Results:")
    print(f"  Frames: {frame_count}, Total balls: {total_balls_detected}")
    print(f"  Avg balls/frame: {avg_balls:.1f}, Max balls: {max_balls}")
    print(f"  Avg FPS: {stats['avg_fps']:.1f}")
    print(f"  Ball types: {stats['ball_types_detected']}")
    
    return stats

def compare_results(single_stats, multi_stats):
    """So sánh kết quả giữa hai phương pháp"""
    
    print(f"\n🏆 COMPARISON RESULTS:")
    print(f"=" * 50)
    
    # Ball detection comparison
    single_balls = single_stats['total_balls']
    multi_balls = multi_stats['total_balls']
    improvement = multi_balls - single_balls
    improvement_percent = (improvement / single_balls * 100) if single_balls > 0 else float('inf')
    
    print(f"🎱 Ball Detection:")
    print(f"  Single-Class: {single_balls} balls")
    print(f"  Multi-Class:  {multi_balls} balls")
    print(f"  Improvement:  +{improvement} balls ({improvement_percent:.1f}%)")
    
    # Average balls per frame
    single_avg = single_stats['avg_balls_per_frame']
    multi_avg = multi_stats['avg_balls_per_frame']
    avg_improvement = multi_avg - single_avg
    
    print(f"\n📈 Average Balls per Frame:")
    print(f"  Single-Class: {single_avg:.1f}")
    print(f"  Multi-Class:  {multi_avg:.1f}")
    print(f"  Improvement:  +{avg_improvement:.1f}")
    
    # Maximum balls per frame
    single_max = single_stats['max_balls_per_frame']
    multi_max = multi_stats['max_balls_per_frame']
    
    print(f"\n🏆 Maximum Balls per Frame:")
    print(f"  Single-Class: {single_max}")
    print(f"  Multi-Class:  {multi_max}")
    print(f"  Improvement:  +{multi_max - single_max}")
    
    # Performance comparison
    single_fps = single_stats['avg_fps']
    multi_fps = multi_stats['avg_fps']
    
    print(f"\n⚡ Performance:")
    print(f"  Single-Class: {single_fps:.1f} FPS")
    print(f"  Multi-Class:  {multi_fps:.1f} FPS")
    print(f"  Difference:   {multi_fps - single_fps:.1f} FPS")
    
    # Ball types detected
    print(f"\n🎯 Ball Types Detected:")
    print(f"  Single-Class: Only class 0 (cue ball)")
    print(f"  Multi-Class:  {len(multi_stats['ball_types_detected'])} types: {multi_stats['ball_types_detected']}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    if improvement > 0:
        print(f"✅ Multi-class detection is {improvement_percent:.1f}% more effective!")
        print(f"✅ Detects {improvement} more balls total")
        print(f"✅ Detects {len(multi_stats['ball_types_detected'])} different ball types")
    else:
        print(f"⚠️ Both methods perform similarly")
    
    return {
        'single_class': single_stats,
        'multi_class': multi_stats,
        'improvement': improvement,
        'improvement_percent': improvement_percent
    }

def save_comparison_results(results, output_file="video_detection_comparison.json"):
    """Lưu kết quả so sánh vào file JSON"""
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare single-class vs multi-class video detection")
    parser.add_argument("--model", default="yolov8m.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    parser.add_argument("--output", default="video_detection_comparison.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    print(f"🎬 Video Detection Comparison")
    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    print(f"Max frames: {args.max_frames}")
    print(f"=" * 50)
    
    # Process with single class
    print(f"\n🔍 Step 1: Single-Class Detection")
    single_stats = process_video_single_class(args.model, args.video, args.max_frames)
    
    if single_stats is None:
        print("❌ Failed to process video with single-class detection")
        return
    
    # Process with multi class
    print(f"\n🔍 Step 2: Multi-Class Detection")
    multi_stats = process_video_multi_class(args.model, args.video, args.max_frames)
    
    if multi_stats is None:
        print("❌ Failed to process video with multi-class detection")
        return
    
    # Compare results
    comparison_results = compare_results(single_stats, multi_stats)
    
    # Save results
    save_comparison_results(comparison_results, args.output)
    
    print(f"\n✅ Comparison completed successfully!")

if __name__ == "__main__":
    main() 
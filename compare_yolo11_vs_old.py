#!/usr/bin/env python3
"""
Compare YOLOv11 vs old model performance
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

def test_model_performance(model_path, video_path, max_frames=500, conf=0.1):
    """Test model performance on video"""
    
    print(f"🤖 Testing model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Statistics
    frame_count = 0
    total_detections = 0
    detection_counts = []
    inference_times = []
    
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
        if frame_count % 100 == 0:
            progress = (frame_count / min(total_frames, max_frames if max_frames else total_frames)) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_count}")
        
        # Run detection
        start_time = time.time()
        results = model(frame, conf=float(conf), verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detections = len(boxes)
                total_detections += detections
                detection_counts.append(detections)
            else:
                detection_counts.append(0)
        else:
            detection_counts.append(0)
    
    # Cleanup
    cap.release()
    
    # Calculate statistics
    avg_detections = np.mean(detection_counts) if detection_counts else 0
    max_detections = max(detection_counts) if detection_counts else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
    
    return {
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'avg_detections_per_frame': avg_detections,
        'max_detections_per_frame': max_detections,
        'avg_inference_time': avg_inference_time,
        'avg_fps': avg_fps
    }

def main():
    print("🏆 YOLOv11 vs Old Model Comparison")
    print("=" * 50)
    
    video_path = "8ball.mp4"
    max_frames = 500
    conf = 0.01  # Very low confidence to see all detections
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    # Test YOLOv11 model
    yolo11_model = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    
    if Path(yolo11_model).exists():
        print(f"\n🎯 Testing YOLOv11 model...")
        yolo11_results = test_model_performance(yolo11_model, video_path, max_frames, conf)
        
        if yolo11_results:
            print(f"\n📊 YOLOv11 Results:")
            print(f"  🎬 Frames processed: {yolo11_results['frames_processed']}")
            print(f"  🎯 Total detections: {yolo11_results['total_detections']}")
            print(f"  📈 Average detections per frame: {yolo11_results['avg_detections_per_frame']:.1f}")
            print(f"  🏆 Maximum detections per frame: {yolo11_results['max_detections_per_frame']}")
            print(f"  ⏱️ Average inference time: {yolo11_results['avg_inference_time']:.3f}s")
            print(f"  🎯 Average FPS: {yolo11_results['avg_fps']:.1f}")
    else:
        print(f"❌ YOLOv11 model not found: {yolo11_model}")
        yolo11_results = None
    
    # Test old model (if exists)
    old_model = "models/billiards_model.pt"
    
    if Path(old_model).exists():
        print(f"\n🎯 Testing Old Model...")
        old_results = test_model_performance(old_model, video_path, max_frames, conf)
        
        if old_results:
            print(f"\n📊 Old Model Results:")
            print(f"  🎬 Frames processed: {old_results['frames_processed']}")
            print(f"  🎯 Total detections: {old_results['total_detections']}")
            print(f"  📈 Average detections per frame: {old_results['avg_detections_per_frame']:.1f}")
            print(f"  🏆 Maximum detections per frame: {old_results['max_detections_per_frame']}")
            print(f"  ⏱️ Average inference time: {old_results['avg_inference_time']:.3f}s")
            print(f"  🎯 Average FPS: {old_results['avg_fps']:.1f}")
    else:
        print(f"❌ Old model not found: {old_model}")
        old_results = None
    
    # Comparison
    if yolo11_results and old_results:
        print(f"\n🏆 COMPARISON RESULTS:")
        print("=" * 50)
        
        # Detections comparison
        detection_diff = yolo11_results['total_detections'] - old_results['total_detections']
        detection_ratio = yolo11_results['total_detections'] / old_results['total_detections'] if old_results['total_detections'] > 0 else float('inf')
        
        print(f"🎯 Total Detections:")
        print(f"  YOLOv11: {yolo11_results['total_detections']}")
        print(f"  Old Model: {old_results['total_detections']}")
        print(f"  Difference: {detection_diff:+d}")
        print(f"  Ratio: {detection_ratio:.2f}x")
        
        # Performance comparison
        fps_diff = yolo11_results['avg_fps'] - old_results['avg_fps']
        fps_ratio = yolo11_results['avg_fps'] / old_results['avg_fps'] if old_results['avg_fps'] > 0 else float('inf')
        
        print(f"\n⚡ Performance:")
        print(f"  YOLOv11: {yolo11_results['avg_fps']:.1f} FPS")
        print(f"  Old Model: {old_results['avg_fps']:.1f} FPS")
        print(f"  Difference: {fps_diff:+.1f} FPS")
        print(f"  Ratio: {fps_ratio:.2f}x")
        
        # Conclusion
        print(f"\n🎉 CONCLUSION:")
        if detection_ratio > 1:
            print(f"✅ YOLOv11 detects {detection_ratio:.2f}x more objects!")
        else:
            print(f"❌ YOLOv11 detects {detection_ratio:.2f}x fewer objects")
            
        if fps_ratio > 1:
            print(f"✅ YOLOv11 is {fps_ratio:.2f}x faster!")
        else:
            print(f"❌ YOLOv11 is {fps_ratio:.2f}x slower")
    
    print(f"\n✅ Comparison completed!")

if __name__ == "__main__":
    main() 
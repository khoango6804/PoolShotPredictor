#!/usr/bin/env python3
"""
Test Improved Multi-Ball Detector
Demo script để test detector cải tiến
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time

# Add src to path
import sys
sys.path.append('src')

from src.models.improved_ball_detector import ImprovedMultiObjectDetector

def test_single_image(model_path, image_path, output_path="test_results"):
    """Test detector trên một hình ảnh"""
    
    print(f"🎯 Testing improved detector...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    
    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Không thể load image: {image_path}")
        return
    
    print(f"Image size: {frame.shape}")
    
    # Test different configurations
    configs = [
        {
            'name': 'Default',
            'config': None
        },
        {
            'name': 'Optimized for Multi-Ball',
            'config': {
                'confidence': 0.15,
                'iou': 0.2,
                'max_detections': 100,
                'enable_size_filtering': False,
                'enable_overlap_filtering': False
            }
        },
        {
            'name': 'High Sensitivity',
            'config': {
                'confidence': 0.1,
                'iou': 0.1,
                'max_detections': 150,
                'enable_size_filtering': False,
                'enable_overlap_filtering': False
            }
        }
    ]
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for config_info in configs:
        print(f"\n🔍 Testing: {config_info['name']}")
        
        # Initialize detector
        detector = ImprovedMultiObjectDetector(model_path, config_info['config'])
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Get statistics
        stats = detector.get_detection_stats(detections)
        
        print(f"  ⏱️ Inference time: {inference_time:.3f}s")
        print(f"  🎱 Balls detected: {stats['ball_count']}")
        print(f"  📊 Total detections: {stats['total_detections']}")
        print(f"  📈 Avg ball confidence: {stats['avg_ball_confidence']:.3f}")
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame.copy(), detections)
        
        # Save result
        output_file = output_dir / f"{config_info['name'].replace(' ', '_')}_result.jpg"
        cv2.imwrite(str(output_file), annotated_frame)
        print(f"  💾 Saved: {output_file}")
        
        # Store results
        results.append({
            'name': config_info['name'],
            'detections': detections,
            'stats': stats,
            'inference_time': inference_time,
            'output_file': output_file
        })
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['stats']['ball_count'])
    
    print(f"\n🏆 BEST CONFIGURATION: {best_result['name']}")
    print(f"  🎱 Balls detected: {best_result['stats']['ball_count']}")
    print(f"  📈 Avg confidence: {best_result['stats']['avg_ball_confidence']:.3f}")
    print(f"  ⏱️ Inference time: {best_result['inference_time']:.3f}s")
    
    return results

def test_video(model_path, video_path, output_path="test_results", max_frames=100):
    """Test detector trên video"""
    
    print(f"🎬 Testing improved detector on video...")
    print(f"Video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không thể open video: {video_path}")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize detector with optimized config
    optimized_config = {
        'confidence': 0.15,
        'iou': 0.2,
        'max_detections': 100,
        'enable_size_filtering': False,
        'enable_overlap_filtering': False
    }
    
    detector = ImprovedMultiObjectDetector(model_path, optimized_config)
    
    # Setup video writer
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    output_video = output_dir / "improved_detection_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Statistics
    frame_count = 0
    total_balls_detected = 0
    ball_counts = []
    inference_times = []
    
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
            progress = (frame_count / min(total_frames, max_frames)) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_count}")
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Get statistics
        stats = detector.get_detection_stats(detections)
        
        # Update statistics
        total_balls_detected += stats['ball_count']
        ball_counts.append(stats['ball_count'])
        inference_times.append(inference_time)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Add statistics text
        cv2.putText(annotated_frame, f"Balls: {stats['ball_count']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {1/inference_time:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame
        out.write(annotated_frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print final statistics
    avg_balls = np.mean(ball_counts) if ball_counts else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    max_balls = max(ball_counts) if ball_counts else 0
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  🎬 Frames processed: {frame_count}")
    print(f"  🎱 Total balls detected: {total_balls_detected}")
    print(f"  📈 Average balls per frame: {avg_balls:.1f}")
    print(f"  🏆 Maximum balls in single frame: {max_balls}")
    print(f"  ⏱️ Average inference time: {avg_inference_time:.3f}s")
    print(f"  🎯 Average FPS: {1/avg_inference_time:.1f}")
    print(f"  💾 Output video: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Test improved multi-ball detector")
    parser.add_argument("--model", default="models/billiards_model.pt", help="Path to YOLO model")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output", default="test_results", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process for video")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Check if input is image or video
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        test_single_image(args.model, args.input, args.output)
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        test_video(args.model, args.input, args.output, args.max_frames)
    else:
        print(f"❌ Unsupported file format: {input_path.suffix}")

if __name__ == "__main__":
    main() 
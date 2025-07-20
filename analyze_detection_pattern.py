#!/usr/bin/env python3
"""
Analyze detection pattern across frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_detection_pattern(model_path, video_path, max_frames=2000, conf=0.01):
    """Analyze detection pattern across frames"""
    
    print(f"🔍 Analyzing detection pattern...")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Statistics
    frame_count = 0
    detection_counts = []
    frame_numbers = []
    
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
        if frame_count % 200 == 0:
            progress = (frame_count / max_frames) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_count}")
        
        # Run detection
        results = model(frame, conf=float(conf), verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detections = len(boxes)
            else:
                detections = 0
        else:
            detections = 0
        
        detection_counts.append(detections)
        frame_numbers.append(frame_count)
        
        # Print detailed info for frames with detections
        if detections > 0:
            print(f"Frame {frame_count}: {detections} detections")
    
    # Cleanup
    cap.release()
    
    # Analyze patterns
    total_detections = sum(detection_counts)
    avg_detections = np.mean(detection_counts)
    max_detections = max(detection_counts)
    
    # Find frames with detections
    frames_with_detections = [i+1 for i, count in enumerate(detection_counts) if count > 0]
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"  🎬 Total frames: {frame_count}")
    print(f"  🎯 Total detections: {total_detections}")
    print(f"  📈 Average detections per frame: {avg_detections:.1f}")
    print(f"  🏆 Maximum detections in single frame: {max_detections}")
    print(f"  📋 Frames with detections: {len(frames_with_detections)}")
    
    if frames_with_detections:
        print(f"  🎯 First detection at frame: {frames_with_detections[0]}")
        print(f"  🎯 Last detection at frame: {frames_with_detections[-1]}")
        print(f"  📊 Detection range: {frames_with_detections[-1] - frames_with_detections[0]} frames")
        
        # Analyze detection clusters
        clusters = []
        current_cluster = [frames_with_detections[0]]
        
        for i in range(1, len(frames_with_detections)):
            if frames_with_detections[i] - frames_with_detections[i-1] <= 10:  # Within 10 frames
                current_cluster.append(frames_with_detections[i])
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [frames_with_detections[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        print(f"  🔗 Detection clusters: {len(clusters)}")
        for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
            print(f"    Cluster {i+1}: frames {cluster[0]}-{cluster[-1]} ({len(cluster)} frames)")
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 8))
        
        # Plot detection counts
        plt.subplot(2, 1, 1)
        plt.plot(frame_numbers, detection_counts, 'b-', alpha=0.7)
        plt.fill_between(frame_numbers, detection_counts, alpha=0.3)
        plt.title('Detection Counts by Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Detections')
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative detections
        plt.subplot(2, 1, 2)
        cumulative = np.cumsum(detection_counts)
        plt.plot(frame_numbers, cumulative, 'r-', linewidth=2)
        plt.title('Cumulative Detections')
        plt.xlabel('Frame Number')
        plt.ylabel('Cumulative Detections')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detection_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Analysis plot saved to: detection_analysis.png")
        
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
    
    return {
        'frame_count': frame_count,
        'total_detections': total_detections,
        'avg_detections': avg_detections,
        'max_detections': max_detections,
        'frames_with_detections': frames_with_detections,
        'detection_counts': detection_counts,
        'frame_numbers': frame_numbers
    }

def main():
    print("🔍 Detection Pattern Analysis")
    print("=" * 40)
    
    model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    video_path = "8ball.mp4"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    # Analyze pattern
    results = analyze_detection_pattern(model_path, video_path, max_frames=2000, conf=0.01)
    
    if results:
        print(f"\n🎉 Analysis completed!")
        print(f"📊 Check 'detection_analysis.png' for visualization")

if __name__ == "__main__":
    main() 
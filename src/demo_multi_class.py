#!/usr/bin/env python3
"""
Demo script for Multi-Class Billiards Detection System
"""

import cv2
import numpy as np
import time
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ball_detector import MultiObjectDetector
from src.models.pocket_detector import PocketDetector
from src.config.config import CLASSES, CLASS_COLORS

def create_demo_image():
    """Create a demo billiards table image"""
    # Create green table background
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)  # Forest green
    
    # Add table border
    cv2.rectangle(img, (50, 50), (1230, 670), (0, 100, 0), 50)
    
    # Add pockets (6 corners)
    pockets = [
        (50, 50), (640, 30), (1230, 50),  # Top
        (50, 670), (640, 690), (1230, 670)  # Bottom
    ]
    for x, y in pockets:
        cv2.circle(img, (x, y), 25, (0, 0, 0), -1)
    
    # Add balls (random positions)
    ball_positions = [
        (300, 200), (400, 300), (500, 250), (600, 350),
        (350, 450), (450, 500), (550, 400), (700, 300)
    ]
    for x, y in ball_positions:
        cv2.circle(img, (x, y), 15, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 15, (0, 0, 0), 2)
    
    # Add cue stick
    cv2.line(img, (200, 400), (400, 350), (139, 69, 19), 8)
    
    return img

def run_demo():
    """Run the multi-class detection demo"""
    print("=== Multi-Class Billiards Detection Demo ===")
    print("Classes:", list(CLASSES.values()))
    print("Colors:", CLASS_COLORS)
    
    # Initialize detectors
    detector = MultiObjectDetector()
    pocket_detector = PocketDetector()
    
    # Create demo image
    print("\nCreating demo image...")
    demo_img = create_demo_image()
    
    # Run detection
    print("Running detection...")
    start_time = time.time()
    
    detections = detector.detect(demo_img)
    detections = detector.filter_overlapping_detections(detections)
    
    # Update pocket detector
    pocket_events = pocket_detector.update(detections, demo_img)
    
    processing_time = time.time() - start_time
    
    # Draw results
    result_img = detector.draw_detections(demo_img, detections)
    
    if pocket_events:
        result_img = pocket_detector.draw_pocket_events(result_img, pocket_events)
    
    # Add statistics overlay
    result_img = add_demo_statistics(result_img, detections, pocket_events, processing_time)
    
    # Display results
    print(f"\nDetection Results:")
    print(f"Processing time: {processing_time:.3f}s")
    print(f"Total detections: {len(detections)}")
    
    # Count by class
    class_counts = {}
    for det in detections:
        class_id = det[5]
        class_name = CLASSES.get(class_id, f"class_{class_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("Detections by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    print(f"Pocket events: {len(pocket_events)}")
    
    # Show image
    try:
        cv2.imshow('Multi-Class Detection Demo', result_img)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        out_path = "demo_result.jpg"
        cv2.imwrite(out_path, result_img)
        print(f"Không thể mở cửa sổ hiển thị. Đã lưu kết quả ra {out_path}. Mở file này để xem ảnh kết quả.")

def add_demo_statistics(img, detections, pocket_events, processing_time):
    """Add statistics overlay to demo image"""
    # Create overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Add text
    y_offset = 30
    cv2.putText(img, "Multi-Class Detection Demo", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    cv2.putText(img, f"Processing time: {processing_time:.3f}s", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    
    cv2.putText(img, f"Total detections: {len(detections)}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    
    # Count by class
    class_counts = {}
    for det in detections:
        class_id = det[5]
        class_name = CLASSES.get(class_id, f"class_{class_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        # Find class_id from class_name
        class_id = None
        for cid, cname in CLASSES.items():
            if cname == class_name:
                class_id = cid
                break
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255)) if class_id is not None else (255, 255, 255)
        cv2.putText(img, f"{class_name}: {count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    cv2.putText(img, f"Pocket events: {len(pocket_events)}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return img

def test_with_video(video_path):
    """Test with a video file"""
    print(f"Testing with video: {video_path}")
    
    # Initialize detectors
    detector = MultiObjectDetector()
    pocket_detector = PocketDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    total_detections = 0
    total_pocket_events = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        detections = detector.detect(frame)
        detections = detector.filter_overlapping_detections(detections)
        
        # Update pocket detector
        pocket_events = pocket_detector.update(detections, frame)
        
        # Draw results
        result_frame = detector.draw_detections(frame, detections)
        
        if pocket_events:
            result_frame = pocket_detector.draw_pocket_events(result_frame, pocket_events)
        
        # Add statistics
        result_frame = add_video_statistics(result_frame, frame_count, detections, pocket_events)
        
        # Display
        cv2.imshow('Video Detection Demo', result_frame)
        
        # Count statistics
        total_detections += len(detections)
        total_pocket_events += len(pocket_events)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print(f"\nVideo Analysis Complete:")
    print(f"Total frames: {frame_count}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
    print(f"Total pocket events: {total_pocket_events}")
    print(f"Pocket events per frame: {total_pocket_events/frame_count:.3f}")

def add_video_statistics(img, frame_count, detections, pocket_events):
    """Add statistics overlay to video frame"""
    # Create overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Add text
    y_offset = 30
    cv2.putText(img, f"Frame: {frame_count}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    cv2.putText(img, f"Detections: {len(detections)}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    cv2.putText(img, f"Pocket events: {len(pocket_events)}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    
    # Show class breakdown
    class_counts = {}
    for det in detections:
        class_id = det[5]
        class_name = CLASSES.get(class_id, f"class_{class_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        # Find class_id from class_name
        class_id = None
        for cid, cname in CLASSES.items():
            if cname == class_name:
                class_id = cid
                break
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255)) if class_id is not None else (255, 255, 255)
        cv2.putText(img, f"{class_name}: {count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Multi-Class Billiards Detection Demo')
    parser.add_argument('--video', '-v', type=str, help='Video file to test with')
    parser.add_argument('--demo', '-d', action='store_true', help='Run with demo image')
    
    args = parser.parse_args()
    
    if args.video:
        test_with_video(args.video)
    else:
        run_demo()

if __name__ == "__main__":
    main() 
import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import json

from models.ball_detector import MultiObjectDetector
from models.pocket_detector import PocketDetector
from models.ball_tracker import BallTracker
from config.config import CLASSES, CLASS_COLORS

class MultiObjectDetectionSystem:
    def __init__(self, model_path=None):
        """
        Initialize the multi-object detection system
        
        Args:
            model_path: Path to custom trained model (optional)
        """
        self.detector = MultiObjectDetector()
        self.pocket_detector = PocketDetector()
        self.ball_tracker = BallTracker()
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Detection history
        self.detection_history = []
        
    def process_video(self, video_path, output_path=None, show_display=True):
        """
        Process video file with multi-object detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_display: Whether to show real-time display
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add statistics overlay
            processed_frame = self.add_statistics_overlay(processed_frame)
            
            # Write to output video
            if writer:
                writer.write(processed_frame)
            
            # Display frame
            if show_display:
                cv2.imshow('Multi-Object Detection', processed_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Save detection results
        self.save_detection_results(output_path.replace('.mp4', '_results.json') if output_path else 'detection_results.json')
        
        print("Processing completed!")
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with detections and annotations
        """
        self.frame_count += 1
        
        # Run detection
        detections = self.detector.detect(frame)
        
        # Filter overlapping detections
        detections = self.detector.filter_overlapping_detections(detections)
        
        # Update pocket detector
        pocket_events = self.pocket_detector.update(detections, frame)
        
        # Update ball tracker
        balls = self.detector.get_balls(detections)
        tracked_balls = self.ball_tracker.update(balls)
        
        # Draw detections
        frame = self.detector.draw_detections(frame, detections)
        
        # Draw pocket events
        if pocket_events:
            frame = self.pocket_detector.draw_pocket_events(frame, pocket_events)
        
        # Draw tracked balls
        frame = self.draw_tracked_balls(frame, tracked_balls)
        
        # Store detection history
        self.detection_history.append({
            'frame': self.frame_count,
            'timestamp': time.time(),
            'detections': detections,
            'pocket_events': pocket_events,
            'tracked_balls': tracked_balls
        })
        
        return frame
    
    def draw_tracked_balls(self, frame, tracked_balls):
        """Draw tracked balls with trajectory"""
        for ball_id, ball_info in tracked_balls.items():
            if 'centroids' in ball_info and len(ball_info['centroids']) > 1:
                # Draw trajectory
                for i in range(1, len(ball_info['centroids'])):
                    pt1 = ball_info['centroids'][i-1]
                    pt2 = ball_info['centroids'][i]
                    cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
                
                # Draw current position
                current_pos = ball_info['centroids'][-1]
                cv2.circle(frame, current_pos, 5, (255, 0, 255), -1)
                cv2.putText(frame, f"Ball {ball_id}", 
                           (current_pos[0] + 10, current_pos[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return frame
    
    def add_statistics_overlay(self, frame):
        """Add statistics overlay to frame"""
        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
        
        # Get pocket statistics
        pocket_stats = self.pocket_detector.get_pocket_statistics()
        
        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        y_offset = 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Total Pockets: {pocket_stats.get('total_pockets', 0)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Recent Pockets: {pocket_stats.get('recent_pockets', 0)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Pocket Rate: {pocket_stats.get('pocket_rate', 0):.2f}/s", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def save_detection_results(self, output_path):
        """Save detection results to JSON file"""
        results = {
            'video_info': {
                'total_frames': self.frame_count,
                'processing_time': time.time() - self.start_time,
                'fps': self.fps
            },
            'detection_history': self.detection_history,
            'pocket_statistics': self.pocket_detector.get_pocket_statistics(),
            'classes_detected': list(CLASSES.keys())
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detection results saved to: {output_path}")
    
    def process_camera(self, camera_id=0, show_display=True):
        """
        Process live camera feed
        
        Args:
            camera_id: Camera device ID
            show_display: Whether to show real-time display
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print(f"Processing camera feed (ID: {camera_id})")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add statistics overlay
            processed_frame = self.add_statistics_overlay(processed_frame)
            
            # Display frame
            if show_display:
                cv2.imshow('Multi-Object Detection (Live)', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save detection results
        self.save_detection_results('live_detection_results.json')

def main():
    parser = argparse.ArgumentParser(description='Multi-Object Detection for Billiards')
    parser.add_argument('--input', '-i', type=str, help='Input video file or camera ID (0, 1, etc.)')
    parser.add_argument('--output', '-o', type=str, help='Output video file')
    parser.add_argument('--model', '-m', type=str, help='Custom model path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Initialize system
    system = MultiObjectDetectionSystem(model_path=args.model)
    
    if args.input:
        # Check if input is a number (camera ID) or file path
        if args.input.isdigit():
            system.process_camera(camera_id=int(args.input), show_display=not args.no_display)
        else:
            system.process_video(args.input, args.output, show_display=not args.no_display)
    else:
        # Default to camera 0
        system.process_camera(camera_id=0, show_display=not args.no_display)

if __name__ == "__main__":
    main() 
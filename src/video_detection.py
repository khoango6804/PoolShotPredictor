import cv2
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ball_detector import MultiObjectDetector
from src.models.pocket_detector import PocketDetector
from src.config.config import CLASSES, CLASS_COLORS

class VideoDetector:
    def __init__(self, model_path=None):
        """Initialize video detector"""
        self.detector = MultiObjectDetector(model_path=model_path)
        self.pocket_detector = PocketDetector()
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def process_video(self, input_path, output_path=None, show_display=True):
        """
        Process video with detection
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (optional)
            show_display: Whether to show live display
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_path}")
        
        # Process frames
        print("Processing video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            self.fps_counter += 1
            
            # Calculate FPS
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Run detection
            detections = self.detector.detect(frame)
            
            # Run pocket detection
            pocket_events = self.pocket_detector.update(detections, frame)
            
            # Draw detections
            frame_with_detections = self.detector.draw_detections(frame.copy(), detections)
            
            # Draw pocket events
            if pocket_events:
                frame_with_detections = self.pocket_detector.draw_pocket_events(
                    frame_with_detections, pocket_events)
            
            # Add statistics overlay
            frame_with_detections = self._add_statistics_overlay(frame_with_detections)
            
            # Write frame if output specified
            if writer:
                writer.write(frame_with_detections)
            
            # Show display
            if show_display:
                cv2.imshow("Billiards Detection", frame_with_detections)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)  # Pause
            
            # Print progress
            if self.frame_count % 30 == 0:  # Every 30 frames
                progress = (self.frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_statistics()
        
        print("Video processing completed!")
    
    def _add_statistics_overlay(self, frame):
        """Add statistics overlay to frame"""
        # Get statistics
        stats = self.pocket_detector.get_pocket_statistics()
        
        # Add FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add pocket statistics
        y_offset = 90
        if stats:
            cv2.putText(frame, f"Total Pockets: {stats.get('total_pockets', 0)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            cv2.putText(frame, f"Recent Pockets: {stats.get('recent_pockets', 0)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            pocket_rate = stats.get('pocket_rate', 0)
            cv2.putText(frame, f"Pocket Rate: {pocket_rate:.2f}/min", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _print_final_statistics(self):
        """Print final processing statistics"""
        stats = self.pocket_detector.get_pocket_statistics()
        
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total pocketing events: {stats.get('total_pockets', 0)}")
        print(f"Recent pocketing events: {stats.get('recent_pockets', 0)}")
        print(f"Pocketing rate: {stats.get('pocket_rate', 0):.2f} pockets/minute")
        
        # Print pocket type breakdown
        pocket_counts = stats.get('pocket_type_counts', {})
        if pocket_counts:
            print("\nPocket type breakdown:")
            for pocket_type, count in pocket_counts.items():
                print(f"  {pocket_type}: {count}")
        
        print("="*50)
    
    def save_pocket_events(self, output_path):
        """Save pocket events to file"""
        self.pocket_detector.save_pocket_events(output_path)
        print(f"Pocket events saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Billiards Video Detection')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', help='Path to trained model')
    parser.add_argument('--no-display', action='store_true', 
                       help='Disable live display')
    parser.add_argument('--save-events', help='Save pocket events to file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Initialize detector
    detector = VideoDetector(model_path=args.model)
    
    # Process video
    detector.process_video(
        input_path=args.input,
        output_path=args.output,
        show_display=not args.no_display
    )
    
    # Save pocket events if requested
    if args.save_events:
        detector.save_pocket_events(args.save_events)

if __name__ == "__main__":
    main() 
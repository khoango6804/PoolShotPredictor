"""
Test Multi-Model Detector với video 8ball.mp4
"""

import cv2
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.multi_model_detector import MultiModelDetector

def test_video(video_path="8ball.mp4", output_path="runs/multi_model_test.mp4"):
    """Test multi-model detector on video"""
    
    print("🎯 Testing Multi-Model Detector...")
    print(f"📹 Video: {video_path}")
    
    # Initialize detector
    detector = MultiModelDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("🚀 Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run multi-model detection
        results = detector.detect_all(frame, conf_threshold=0.3)
        
        # Get ball summary for analysis
        ball_summary = detector.get_ball_summary(results)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, results)
        
        # Add comprehensive info overlay
        info_y = 50
        
        # Ball count info
        cv2.putText(annotated_frame, f"🎱 MULTI-MODEL DETECTION", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(annotated_frame, f"Total Balls: {ball_summary['total_balls']}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        cv2.putText(annotated_frame, f"Solids (1-7): {len(ball_summary['solids'])}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        info_y += 25
        
        cv2.putText(annotated_frame, f"Stripes (9-15): {len(ball_summary['stripes'])}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        info_y += 25
        
        # Special balls
        if ball_summary['cue_ball']:
            cv2.putText(annotated_frame, "✅ Cue Ball Detected", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            info_y += 20
        
        if ball_summary['eight_ball']:
            cv2.putText(annotated_frame, "✅ Eight Ball Detected", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Add white background for black text
            cv2.rectangle(annotated_frame, (8, info_y-15), (200, info_y+5), (255, 255, 255), -1)
            cv2.putText(annotated_frame, "✅ Eight Ball Detected", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            info_y += 20
        
        # Detection counts
        cv2.putText(annotated_frame, f"Tables: {len(results['tables'])}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        info_y += 20
        
        cv2.putText(annotated_frame, f"Pockets: {len(results['pockets'])}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        info_y += 20
        
        # Frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                   (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        progress = frame_count / total_frames
        bar_width = 300
        bar_height = 10
        bar_x = width - bar_width - 20
        bar_y = 50
        
        # Background
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        # Progress
        cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        
        # Write frame
        out.write(annotated_frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"🎬 Frame {frame_count}/{total_frames} ({progress*100:.1f}%) - "
                  f"Balls: {ball_summary['total_balls']}")
        
        # Print detailed ball info when detected
        if ball_summary['total_balls'] > 0:
            solid_numbers = [ball['ball_number'] for ball in ball_summary['solids'] 
                           if ball.get('ball_number')]
            stripe_numbers = [ball['ball_number'] for ball in ball_summary['stripes'] 
                            if ball.get('ball_number')]
            
            if solid_numbers or stripe_numbers:
                print(f"🎱 Frame {frame_count}: Solids={solid_numbers}, Stripes={stripe_numbers}")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📹 Output saved: {output_path}")
    print(f"🎬 Processed {frame_count} frames")

def main():
    """Main function"""
    video_path = "8ball.mp4"
    output_path = "runs/multi_model_analysis.mp4"
    
    # Create output directory
    Path("runs").mkdir(exist_ok=True)
    
    test_video(video_path, output_path)

if __name__ == "__main__":
    main() 
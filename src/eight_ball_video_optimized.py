#!/usr/bin/env python3
"""
Optimized 8-Ball Pool Video Analysis
Uses high resolution and optimized settings for maximum ball detection
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.eight_ball_game import EightBallGame

def main():
    """Main function for optimized video analysis"""
    parser = argparse.ArgumentParser(description='Optimized 8-Ball Pool Video Analysis')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--model_path', default='runs/ball_classification/yolov8m_correct/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--output_dir', default='runs/optimized_analysis', help='Output directory')
    parser.add_argument('--headless', action='store_true', help='Run without display')
    parser.add_argument('--confidence', type=float, default=0.05, help='Detection confidence threshold')
    parser.add_argument('--resolution', type=int, default=1280, help='Detection resolution')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 OPTIMIZED 8-BALL POOL ANALYSIS")
    print("=" * 60)
    print(f"📹 Video: {args.video_path}")
    print(f"🤖 Model: {args.model_path}")
    print(f"📊 Confidence: {args.confidence}")
    print(f"🔍 Resolution: {args.resolution}x{args.resolution}")
    print(f"💾 Output: {output_dir}")
    
    # Load model
    print("\n🔄 Loading model...")
    model = YOLO(args.model_path)
    
    # Initialize game
    game = EightBallGame()
    game.start_game()
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📺 Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    video_name = Path(args.video_path).stem
    output_video = output_dir / f"{video_name}_optimized_analysis.mp4"
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Processing variables
    frame_count = 0
    detection_stats = {
        'total_detections': 0,
        'unique_balls_detected': set(),
        'avg_balls_per_frame': [],
        'confidence_stats': []
    }
    
    print(f"\n🎬 Starting analysis...")
    print("Press 'q' to quit, 'p' to pause")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process max frames limit
            if args.max_frames and frame_count > args.max_frames:
                break
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_count}")
            
            # Run detection with optimized settings
            results = model(frame, 
                          imgsz=args.resolution,
                          conf=args.confidence,
                          verbose=False)
            
            # Process detections
            detected_balls = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    
                    for i, (cls, conf, box) in enumerate(zip(classes, confidences, xyxy)):
                        ball_class = int(cls)
                        detected_balls.append({
                            'class': ball_class,
                            'confidence': float(conf),
                            'bbox': box.tolist()
                        })
                        
                        # Update stats
                        detection_stats['unique_balls_detected'].add(ball_class)
                        detection_stats['confidence_stats'].append(float(conf))
                    
                    detection_stats['total_detections'] += len(detected_balls)
                    detection_stats['avg_balls_per_frame'].append(len(detected_balls))
            
            # Update game state
            ball_positions = {}
            for ball in detected_balls:
                ball_class = ball['class']
                # Convert bbox to center position
                bbox = ball['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                ball_positions[ball_class] = (center_x, center_y)
            
            # Update game logic (simplified - just track detected balls)
            # game.update_from_detection(ball_positions)  # Method may not exist
            
            # Create visualization
            annotated_frame = frame.copy()
            
            # Draw detections
            if results and len(results) > 0:
                annotated_frame = results[0].plot()
            
            # Add game info overlay
            game_info = [
                f"Frame: {frame_count}/{total_frames}",
                f"Balls detected: {len(detected_balls)}",
                f"Unique balls: {len(detection_stats['unique_balls_detected'])}",
                f"Game state: {game.game_state}",
                f"Current player: {game.current_player.name}",
                f"Score: P1={game.player1.score}, P2={game.player2.score}",
            ]
            
            y_offset = 30
            for info in game_info:
                cv2.putText(annotated_frame, info, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
            
            # Display ball list
            if detected_balls:
                ball_info = f"Detected: {[ball['class'] for ball in detected_balls]}"
                cv2.putText(annotated_frame, ball_info, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Display if not headless
            if not args.headless:
                cv2.imshow('Optimized 8-Ball Analysis', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            
            # Check for game over
            if game.game_state == "game_over":
                print(f"\nGame Over! Winner: {game.current_player.name}")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if not args.headless:
            cv2.destroyAllWindows()
    
    # Calculate final stats
    avg_balls = np.mean(detection_stats['avg_balls_per_frame']) if detection_stats['avg_balls_per_frame'] else 0
    avg_confidence = np.mean(detection_stats['confidence_stats']) if detection_stats['confidence_stats'] else 0
    unique_balls = sorted(list(detection_stats['unique_balls_detected']))
    
    # Save results
    results_data = {
        "video_info": {
            "path": args.video_path,
            "frames_processed": frame_count,
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "detection_resolution": f"{args.resolution}x{args.resolution}",
            "confidence_threshold": args.confidence
        },
        "detection_stats": {
            "total_detections": detection_stats['total_detections'],
            "unique_balls_detected": unique_balls,
            "unique_ball_count": len(unique_balls),
            "avg_balls_per_frame": float(avg_balls),
            "avg_confidence": float(avg_confidence),
            "missing_balls": [i for i in range(16) if i not in unique_balls]
        },
        "final_game_state": game.get_game_state(),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    # Save results
    results_file = output_dir / f"{video_name}_optimized_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📹 Output video: {output_video}")
    print(f"📊 Results: {results_file}")
    print(f"🎱 Unique balls detected: {len(unique_balls)}/16")
    print(f"🎯 Balls found: {unique_balls}")
    print(f"❌ Missing balls: {results_data['detection_stats']['missing_balls']}")
    print(f"📈 Avg balls per frame: {avg_balls:.1f}")
    print(f"🔢 Avg confidence: {avg_confidence:.3f}")

if __name__ == "__main__":
    main() 
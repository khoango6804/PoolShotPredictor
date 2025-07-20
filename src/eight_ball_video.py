"""
8-Ball Pool Video Processing
Process real videos with 8-ball pool game logic and detection
"""

import cv2
import torch
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.ball_detector import MultiObjectDetector
from models.eight_ball_game import EightBallGame
from config.eight_ball_config import *

def draw_enhanced_detections(frame, detections, game):
    """Draw enhanced detections with 8-ball specific information"""
    annotated_frame = frame.copy()
    
    # Draw basic detections first
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        class_id = int(class_id)
        
        # Get class info with fallback for unknown classes
        if class_id in EIGHT_BALL_CLASSES and class_id in EIGHT_BALL_COLORS:
            class_name = EIGHT_BALL_CLASSES[class_id]
            color = EIGHT_BALL_COLORS[class_id]
        elif class_id in EIGHT_BALL_CLASSES:
            class_name = EIGHT_BALL_CLASSES[class_id]
            color = (255, 255, 255)  # White for known classes without color
        else:
            # Handle unknown classes (from original model)
            class_name = f"class_{class_id}"
            color = (255, 255, 255)  # White for unknown classes
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Enhanced labels for balls
        if class_id in [12, 13, 14, 15]:  # Ball types
            label = f"{class_name}: {conf:.2f}"
            if class_id == 12:  # Cue ball
                label += " (CUE)"
            elif class_id == 13:  # Eight ball  
                label += " (8)"
            elif class_id == 14:  # Solid
                label += " (SOLID)"
            elif class_id == 15:  # Stripe
                label += " (STRIPE)"
        else:
            label = f"{class_name}: {conf:.2f}"
        
        # Draw label
        cv2.putText(annotated_frame, label, 
                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
        
        # Draw center point for balls and pockets
        if class_id in [12, 13, 14, 15] or class_id in range(2, 12):  # balls or pockets
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Draw game overlay
    annotated_frame = game.draw_game_overlay(annotated_frame)
    
    return annotated_frame

def main():
    parser = argparse.ArgumentParser(description='8-Ball Pool Video Analysis')
    parser.add_argument('video_path', help='Path to video file (use 0 for webcam)')
    parser.add_argument('--model_path', default='models/billiards_model.pt', help='Path to model')
    parser.add_argument('--headless', action='store_true', help='Run without display (save output only)')
    parser.add_argument('--output_dir', default='runs/eight_ball', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector and game
    detector = MultiObjectDetector(args.model_path)
    game = EightBallGame()
    game.start_game()
    
    # Video setup
    if args.video_path == '0':
        cap = cv2.VideoCapture(0)
        video_name = 'webcam'
    else:
        cap = cv2.VideoCapture(args.video_path)
        video_name = Path(args.video_path).stem
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Game started! Player {game.current_player.name} to break.")
    print("Processing video with 8-ball pool game logic...")
    
    if not args.headless:
        print("\nControls:")
        print("- 'q': Quit")
        print("- 'p': Pause/unpause")
        print("- 's': Save current game state")
        print("- '1-6': Call pocket for 8-ball")
        print("- 'r': Reset game")
    
    # Video writer for output
    output_video_path = output_dir / f"{video_name}_8ball_analysis.mp4"
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_count = 0
    paused = False
    game_log = []
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                detections = detector.detect(frame)
                
                # Update game state with detections
                game_events = game.update_detections(detections, frame)
                
                # Log events
                if game_events and 'events' in game_events:
                    frame_log = {
                        'frame': frame_count,
                        'events': game_events['events'],
                        'game_state': game.get_game_state(),
                        'timestamp': datetime.now().isoformat()
                    }
                    game_log.append(frame_log)
                    
                    # Print important events
                    for event_type, event_data in game_events['events'].items():
                        if event_type in ['balls_pocketed', 'fouls', 'game_over']:
                            print(f"Frame {frame_count}: {event_type.replace('_', ' ').title()}: {event_data}")
                
                # Draw enhanced visualization
                annotated_frame = draw_enhanced_detections(frame.copy(), detections, game)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Display progress
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% - Frame {frame_count}")
                
                if not args.headless:
                    cv2.imshow('8-Ball Pool Analysis', annotated_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif key == ord('s'):
                        game_state_file = output_dir / f"game_state_frame_{frame_count}.json"
                        with open(game_state_file, 'w') as f:
                            json.dump(game.get_game_state(), f, indent=2)
                        print(f"Game state saved to {game_state_file}")
                    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
                        pocket_num = chr(key)
                        game.call_pocket(f"pocket_{pocket_num}")
                        print(f"Pocket {pocket_num} called for 8-ball")
                    elif key == ord('r'):
                        # Reset game by creating new instance
                        game = EightBallGame()
                        game.start_game()
                        print("Game reset")
                
                # Check if game is over
                if game.game_state == GameState.GAME_OVER:
                    state = game.get_game_state()
                    winner = state['player1']['name'] if state['player1']['score'] > state['player2']['score'] else state['player2']['name']
                    print(f"\nGame Over! {winner} wins!")
                    break
            else:
                # Paused - just check for key presses
                if not args.headless:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = False
                        print("Resumed")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Save final results
        results_file = output_dir / f"{video_name}_results.json"
        final_game_state = game.get_game_state()
        final_results = {
            'video_info': {
                'path': str(args.video_path),
                'frames_processed': frame_count,
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f"{width}x{height}"
            },
            'final_game_state': final_game_state,
            'game_log': game_log,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nAnalysis complete!")
        print(f"- Processed {frame_count} frames")
        print(f"- Output video: {output_video_path}")
        print(f"- Results saved: {results_file}")
        print(f"- Game events logged: {len(game_log)}")
        
        # Print final game summary
        state = final_game_state
        print(f"\nFinal Game State:")
        print(f"- State: {state['game_state']}")
        print(f"- Current Player: {state['current_player']['name']}")
        print(f"- Player 1 Score: {state['player1']['score']}")
        print(f"- Player 2 Score: {state['player2']['score']}")
        
        # Cleanup
        cap.release()
        out.release()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
"""
Demo script for 8-Ball Pool Game System
Tests the game logic and detection integration
"""

import cv2
import numpy as np
import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ball_detector import MultiObjectDetector
from src.models.eight_ball_game import EightBallGame
from src.config.eight_ball_config import (
    EIGHT_BALL_CLASSES, EIGHT_BALL_COLORS, GameState, PlayerType
)

class EightBallDemo:
    """Demo class for 8-ball pool game"""
    
    def __init__(self, model_path=None):
        """Initialize demo with detector and game"""
        self.detector = MultiObjectDetector(model_path=model_path)
        self.game = EightBallGame("Player 1", "Player 2")
        self.frame_count = 0
        
    def create_synthetic_game_scene(self):
        """Create a synthetic 8-ball pool game scene for testing"""
        # Create green billiards table background
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:, :] = (34, 139, 34)  # Forest green
        
        # Draw table border
        cv2.rectangle(img, (50, 50), (1230, 670), (139, 69, 19), 20)  # Brown border
        
        # Draw pockets at standard positions
        pocket_positions = [
            (80, 80),      # TopLeft
            (1200, 80),    # TopRight
            (80, 640),     # BottomLeft
            (1200, 640),   # BottomRight
            (640, 80),     # MediumLeft (top)
            (640, 640)     # MediumRight (bottom)
        ]
        
        for pos in pocket_positions:
            cv2.circle(img, pos, 25, (0, 0, 0), -1)  # Black pockets
        
        # Draw 8-ball pool balls in rack formation
        rack_center_x, rack_center_y = 900, 360
        ball_radius = 12
        
        # Cue ball (separate from rack)
        cue_ball_pos = (400, 360)
        cv2.circle(img, cue_ball_pos, ball_radius, (255, 255, 255), -1)  # White cue ball
        cv2.putText(img, "CUE", (cue_ball_pos[0]-15, cue_ball_pos[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # 8-ball at center of rack
        eight_ball_pos = (rack_center_x, rack_center_y)
        cv2.circle(img, eight_ball_pos, ball_radius, (0, 0, 0), -1)  # Black 8-ball
        cv2.putText(img, "8", (eight_ball_pos[0]-5, eight_ball_pos[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Solid balls (1-7) - Left side of rack
        solid_positions = [
            (rack_center_x-30, rack_center_y-15),  # Position 1
            (rack_center_x-30, rack_center_y+15),  # Position 2
            (rack_center_x-15, rack_center_y-30),  # Position 3
            (rack_center_x-15, rack_center_y+30),  # Position 4
        ]
        
        solid_colors = [(255, 255, 0), (0, 0, 255), (255, 0, 0), (128, 0, 128)]  # Yellow, Blue, Red, Purple
        for i, (pos, color) in enumerate(zip(solid_positions, solid_colors)):
            cv2.circle(img, pos, ball_radius, color, -1)
            cv2.putText(img, str(i+1), (pos[0]-5, pos[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Stripe balls (9-15) - Right side of rack  
        stripe_positions = [
            (rack_center_x+15, rack_center_y-30),  # Position 9
            (rack_center_x+15, rack_center_y+30),  # Position 10
            (rack_center_x+30, rack_center_y-15),  # Position 11
            (rack_center_x+30, rack_center_y+15),  # Position 12
        ]
        
        stripe_colors = [(255, 255, 0), (0, 0, 255), (255, 0, 0), (128, 0, 128)]  # Same base colors but striped
        for i, (pos, color) in enumerate(zip(stripe_positions, stripe_colors)):
            # Draw striped pattern
            cv2.circle(img, pos, ball_radius, (255, 255, 255), -1)  # White base
            cv2.circle(img, pos, ball_radius, color, 3)  # Colored stripe
            cv2.putText(img, str(i+9), (pos[0]-7, pos[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return img
    
    def simulate_detections_from_scene(self, img):
        """
        Simulate YOLO detections based on synthetic scene
        Returns detections in the format expected by the game
        """
        detections = []
        
        # Simulate cue ball detection (class_id: 12)
        cue_ball_pos = (400, 360)
        detections.append([
            cue_ball_pos[0]-15, cue_ball_pos[1]-15,  # x1, y1
            cue_ball_pos[0]+15, cue_ball_pos[1]+15,  # x2, y2
            0.95, 12  # confidence, class_id (cue_ball)
        ])
        
        # Simulate 8-ball detection (class_id: 13)
        eight_ball_pos = (900, 360)
        detections.append([
            eight_ball_pos[0]-15, eight_ball_pos[1]-15,
            eight_ball_pos[0]+15, eight_ball_pos[1]+15,
            0.92, 13  # eight_ball
        ])
        
        # Simulate solid balls (class_id: 14)
        solid_positions = [
            (870, 345), (870, 375), (885, 330), (885, 390)
        ]
        for pos in solid_positions:
            detections.append([
                pos[0]-12, pos[1]-12,
                pos[0]+12, pos[1]+12,
                0.88, 14  # solid_ball
            ])
        
        # Simulate stripe balls (class_id: 15)
        stripe_positions = [
            (915, 330), (915, 390), (930, 345), (930, 375)
        ]
        for pos in stripe_positions:
            detections.append([
                pos[0]-12, pos[1]-12,
                pos[0]+12, pos[1]+12,
                0.87, 15  # stripe_ball
            ])
            
        # Simulate table detection (class_id: 1)
        detections.append([
            50, 50, 1230, 670,  # Table boundaries
            0.99, 1  # table
        ])
        
        # Simulate pocket detections
        pocket_data = [
            ((80, 80), 10),      # TopLeft
            ((1200, 80), 11),    # TopRight
            ((80, 640), 2),      # BottomLeft
            ((1200, 640), 3),    # BottomRight
            ((640, 80), 6),      # MediumLeft
            ((640, 640), 7),     # MediumRight
        ]
        
        for (pos, class_id) in pocket_data:
            detections.append([
                pos[0]-25, pos[1]-25,
                pos[0]+25, pos[1]+25,
                0.85, class_id
            ])
        
        return detections
    
    def draw_enhanced_detections(self, frame, detections):
        """Draw detections with 8-ball specific styling"""
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # Get color and name for this class
            color = EIGHT_BALL_COLORS.get(class_id, (255, 255, 255))
            class_name = EIGHT_BALL_CLASSES.get(class_id, f"class_{class_id}")
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw enhanced label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(frame, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, 
                       (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
            
            # Draw center point for balls and pockets
            if class_id in [12, 13, 14, 15] or class_id in range(2, 12):
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
        return frame
    
    def demo_game_flow(self):
        """Demonstrate complete game flow"""
        print("🎱 8-Ball Pool Demo Starting...")
        print("=" * 50)
        
        # Create synthetic scene
        scene = self.create_synthetic_game_scene()
        
        # Start the game
        self.game.start_game()
        print(f"Game State: {self.game.game_state}")
        print(f"Current Player: {self.game.current_player.name}")
        
        # Simulate some game events
        frame_count = 0
        max_frames = 300  # Demo for 300 frames
        
        while frame_count < max_frames and self.game.game_state != GameState.GAME_OVER:
            frame_count += 1
            
            # Create frame copy for this iteration
            frame = scene.copy()
            
            # Simulate detections
            detections = self.simulate_detections_from_scene(frame)
            
            # Simulate ball movement/pocketing after some frames
            if frame_count == 50:
                print("\n🎯 Simulating break shot...")
                # Remove one solid ball (simulate pocketing)
                detections = [d for d in detections if not (d[5] == 14 and d[0] > 860 and d[0] < 880)]
                
            elif frame_count == 100:
                print("\n🎯 Simulating stripe ball pocketed...")
                # Remove one stripe ball
                detections = [d for d in detections if not (d[5] == 15 and d[0] > 910 and d[0] < 930)]
                
            elif frame_count == 150:
                print("\n🎯 Simulating cue ball pocketed (foul)...")
                # Remove cue ball (simulate scratch)
                detections = [d for d in detections if d[5] != 12]
                
            elif frame_count == 200:
                print("\n🎯 Cue ball replaced (ball in hand)...")
                # Add cue ball back
                detections.append([385, 345, 415, 375, 0.95, 12])
                
            # Update game with detections
            update_info = self.game.update_detections(detections, frame)
            
            # Draw detections
            frame = self.draw_enhanced_detections(frame, detections)
            
            # Draw game overlay
            frame = self.game.draw_game_overlay(frame)
            
            # Add demo info
            cv2.putText(frame, f"Demo Frame: {frame_count}/{max_frames}", 
                       (frame.shape[1]-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("8-Ball Pool Demo", frame)
            
            # Print game updates
            if update_info['balls_pocketed']:
                print(f"  Balls pocketed: {update_info['balls_pocketed']}")
            if update_info['fouls_detected']:
                print(f"  Fouls detected: {update_info['fouls_detected']}")
            if update_info['state_changes']:
                print(f"  State changes: {update_info['state_changes']}")
            
            # Control demo speed
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)  # Pause
            elif key == ord('s'):
                cv2.waitKey(1000)  # Slow motion
        
        # Print final game state
        print("\n" + "=" * 50)
        print("🏁 Demo Complete!")
        print("Final Game State:")
        final_state = self.game.get_game_state()
        for key, value in final_state.items():
            print(f"  {key}: {value}")
        
        cv2.destroyAllWindows()
    
    def test_rule_engine(self):
        """Test the rule engine with various scenarios"""
        print("🧪 Testing Rule Engine...")
        print("-" * 30)
        
        # Test 1: Break shot
        print("Test 1: Break Shot")
        self.game.start_game()
        assert self.game.game_state == GameState.BREAK
        print("✅ Break state initialized correctly")
        
        # Test 2: Player switching
        print("\nTest 2: Player Switching")
        initial_player = self.game.current_player.name
        self.game._switch_players()
        assert self.game.current_player.name != initial_player
        print("✅ Player switching works correctly")
        
        # Test 3: Eight ball rules
        print("\nTest 3: Eight Ball Rules")
        self.game.current_player.balls_remaining = []  # Clear all balls
        can_shoot_eight = self.game.current_player.can_shoot_eight_ball()
        assert can_shoot_eight == True
        print("✅ Eight ball eligibility check works")
        
        # Test 4: Pocket calling
        print("\nTest 4: Pocket Calling")
        self.game.call_pocket("top_left")
        assert self.game.called_pocket == "top_left"
        print("✅ Pocket calling works correctly")
        
        print("\n🎉 All rule engine tests passed!")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='8-Ball Pool Demo')
    parser.add_argument('--model', '-m', help='Path to trained model')
    parser.add_argument('--test-rules', action='store_true', 
                       help='Test rule engine only')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = EightBallDemo(model_path=args.model)
    
    if args.test_rules:
        demo.test_rule_engine()
    else:
        print("🎱 8-Ball Pool Detection & Game Demo")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Slow motion")
        print("-" * 40)
        
        if not args.no_display:
            demo.demo_game_flow()
        else:
            print("Running headless demo...")
            demo.test_rule_engine()

if __name__ == "__main__":
    main() 
"""
8-Ball Pool Game State Tracker and Rule Engine
Manages game state, validates shots, and enforces 8-ball pool rules
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.eight_ball_config import (
    GameState, PlayerType, ShotType, FoulType, WinCondition,
    EIGHT_BALL_RULES, BALL_NUMBERS, EIGHT_BALL_DETECTION,
    POCKET_MAPPING, GAME_TIMING, EIGHT_BALL_CLASSES, EIGHT_BALL_COLORS
)

@dataclass
class Player:
    """Player information"""
    name: str
    player_type: str = PlayerType.NONE
    score: int = 0
    fouls: int = 0
    balls_remaining: List[int] = field(default_factory=list)
    
    def can_shoot_eight_ball(self) -> bool:
        """Check if player can legally shoot 8-ball"""
        return len(self.balls_remaining) == 0

@dataclass
class Shot:
    """Shot information"""
    player: str
    shot_type: str
    start_time: float
    end_time: Optional[float] = None
    cue_ball_start: Optional[Tuple[float, float]] = None
    cue_ball_end: Optional[Tuple[float, float]] = None
    balls_pocketed: List[int] = field(default_factory=list)
    first_ball_hit: Optional[int] = None
    fouls: List[str] = field(default_factory=list)
    is_legal: bool = True
    called_pocket: Optional[str] = None

class EightBallGame:
    """8-Ball Pool Game State Tracker and Rule Engine"""
    
    def __init__(self, player1_name: str = "Player 1", player2_name: str = "Player 2"):
        """Initialize 8-ball pool game"""
        # Players
        self.player1 = Player(name=player1_name, balls_remaining=list(range(1, 8)))
        self.player2 = Player(name=player2_name, balls_remaining=list(range(9, 16)))
        self.current_player = self.player1
        
        # Game state
        self.game_state = GameState.WAITING
        self.shot_count = 0
        self.game_start_time = time.time()
        
        # Ball tracking
        self.balls_on_table = set(range(1, 16))  # 1-15 balls
        self.cue_ball_on_table = True
        self.eight_ball_on_table = True
        
        # Shot tracking
        self.current_shot: Optional[Shot] = None
        self.shot_history: List[Shot] = []
        
        # Game settings
        self.rules = EIGHT_BALL_RULES
        self.called_pocket: Optional[str] = None
        
        # Detection history
        self.detection_history: List[Dict] = []
        
    def start_game(self):
        """Start a new game"""
        self.game_state = GameState.BREAK
        self.current_player = self.player1  # Player 1 breaks
        self._start_shot(ShotType.BREAK)
        print(f"Game started! {self.current_player.name} to break.")
        
    def _start_shot(self, shot_type: str):
        """Start a new shot"""
        self.current_shot = Shot(
            player=self.current_player.name,
            shot_type=shot_type,
            start_time=time.time()
        )
        self.shot_count += 1
        
    def _end_shot(self):
        """End current shot and analyze results"""
        if not self.current_shot:
            return
            
        self.current_shot.end_time = time.time()
        self._analyze_shot()
        self.shot_history.append(self.current_shot)
        self.current_shot = None
        
    def update_detections(self, detections: List, frame: np.ndarray) -> Dict[str, Any]:
        """
        Update game state with new detections
        
        Args:
            detections: List of detection results
            frame: Current frame
            
        Returns:
            Game state update information
        """
        # Store detection history
        detection_data = {
            'timestamp': time.time(),
            'detections': detections,
            'game_state': self.game_state,
            'current_player': self.current_player.name
        }
        self.detection_history.append(detection_data)
        
        # Extract ball information
        balls_detected = self._extract_balls_from_detections(detections)
        pockets_detected = self._extract_pockets_from_detections(detections)
        
        # Update ball positions and states
        update_info = self._update_ball_states(balls_detected, pockets_detected)
        
        # Check for shot completion and rule validation
        if self.current_shot:
            self._check_shot_completion(balls_detected)
            
        return update_info
        
    def _extract_balls_from_detections(self, detections: List) -> Dict[str, List]:
        """Extract ball information from detections"""
        balls = {
            'cue_ball': [],
            'eight_ball': [],
            'solid_balls': [],
            'stripe_balls': [],
            'generic_balls': []
        }
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if class_id == 12:  # cue_ball
                balls['cue_ball'].append({
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
            elif class_id == 13:  # eight_ball
                balls['eight_ball'].append({
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
            elif class_id == 14:  # solid_ball
                balls['solid_balls'].append({
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
            elif class_id == 15:  # stripe_ball
                balls['stripe_balls'].append({
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
            elif class_id == 0:  # generic ball (legacy)
                balls['generic_balls'].append({
                    'position': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
                
        return balls
        
    def _extract_pockets_from_detections(self, detections: List) -> List[Dict]:
        """Extract pocket information from detections"""
        pockets = []
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            if class_id in range(2, 12):  # Pocket classes
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                pocket_type = EIGHT_BALL_CLASSES.get(class_id, f"pocket_{class_id}")
                
                pockets.append({
                    'position': (center_x, center_y),
                    'type': pocket_type,
                    'class_id': class_id,
                    'confidence': conf
                })
                
        return pockets
        
    def _update_ball_states(self, balls_detected: Dict, pockets_detected: List) -> Dict[str, Any]:
        """Update ball states and detect pocketing events"""
        update_info = {
            'balls_pocketed': [],
            'fouls_detected': [],
            'state_changes': []
        }
        
        # Check cue ball
        if not balls_detected['cue_ball'] and self.cue_ball_on_table:
            # Cue ball was pocketed
            self.cue_ball_on_table = False
            update_info['balls_pocketed'].append('cue_ball')
            update_info['fouls_detected'].append(FoulType.CUE_BALL_POCKETED)
            
        elif balls_detected['cue_ball'] and not self.cue_ball_on_table:
            # Cue ball was placed back (ball in hand)
            self.cue_ball_on_table = True
            update_info['state_changes'].append('cue_ball_placed')
            
                 # Check eight ball
        if not balls_detected['eight_ball'] and self.eight_ball_on_table:
            # Eight ball was pocketed
            self.eight_ball_on_table = False
            update_info['balls_pocketed'].append(8)
            self._handle_eight_ball_pocketed(None)
            
        # Check for ball pocketing based on proximity to pockets
        for pocket in pockets_detected:
            self._check_balls_near_pocket(pocket, balls_detected, update_info)
            
        return update_info
        
    def _check_balls_near_pocket(self, pocket: Dict, balls_detected: Dict, update_info: Dict):
        """Check if any balls are near a pocket (pocketed)"""
        pocket_pos = pocket['position']
        threshold = EIGHT_BALL_DETECTION['pocket_distance_threshold']
        
        # Check all ball types
        for ball_type, ball_list in balls_detected.items():
            if ball_type == 'generic_balls':
                continue  # Skip generic balls for now
                
            for ball in ball_list:
                ball_pos = ball['position']
                distance = np.sqrt((ball_pos[0] - pocket_pos[0])**2 + 
                                 (ball_pos[1] - pocket_pos[1])**2)
                
                if distance <= threshold:
                    # Ball is near pocket - consider it pocketed
                    self._process_ball_pocketing(ball_type, pocket, update_info)
                    
    def _process_ball_pocketing(self, ball_type: str, pocket: Dict, update_info: Dict):
        """Process a ball pocketing event"""
        if ball_type == 'cue_ball':
            if 'cue_ball' not in update_info['balls_pocketed']:
                update_info['balls_pocketed'].append('cue_ball')
                update_info['fouls_detected'].append(FoulType.CUE_BALL_POCKETED)
                
        elif ball_type == 'eight_ball':
            if 8 not in update_info['balls_pocketed']:
                update_info['balls_pocketed'].append(8)
                self._handle_eight_ball_pocketed(pocket)
                
        # For solid/stripe balls, we'd need better identification
        # This is a limitation of the current detection system
        
    def _handle_eight_ball_pocketed(self, pocket: Optional[Dict] = None):
        """Handle eight ball pocketing"""
        if self.current_player.can_shoot_eight_ball():
            # Legal eight ball shot
            if pocket and self.called_pocket:
                # Check if called pocket matches
                if POCKET_MAPPING.get(pocket['type']) == self.called_pocket:
                    # Win!
                    self._end_game(WinCondition.EIGHT_BALL_POCKETED)
                else:
                    # Wrong pocket - lose
                    self._end_game(WinCondition.OPPONENT_FOUL_ON_EIGHT, foul=True)
            else:
                # Eight ball pocketed but no called pocket - assume win for now
                self._end_game(WinCondition.EIGHT_BALL_POCKETED)
        else:
            # Eight ball pocketed early - lose
            self._end_game(WinCondition.OPPONENT_FOUL_ON_EIGHT, foul=True)
            
    def _check_shot_completion(self, balls_detected: Dict):
        """Check if current shot is complete"""
        # Simple heuristic: shot is complete when balls stop moving
        # In a real implementation, this would analyze ball velocities
        
        if self.current_shot:
            shot_duration = time.time() - self.current_shot.start_time
            
            # If shot has been going for more than a few seconds, consider it complete
            if shot_duration > 3.0:
                self._end_shot()
                
    def _analyze_shot(self):
        """Analyze completed shot for rule violations"""
        if not self.current_shot:
            return
            
        # Placeholder for shot analysis
        # In a real implementation, this would analyze:
        # - Which ball was hit first
        # - Whether a rail was contacted
        # - Whether the shot was legal according to current game state
        
        self.current_shot.is_legal = True  # Placeholder
        
        # Switch players if no legal ball was pocketed
        if self.current_shot.is_legal and not self.current_shot.balls_pocketed:
            self._switch_players()
            
    def _switch_players(self):
        """Switch current player"""
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
        print(f"Turn: {self.current_player.name}")
        
    def _end_game(self, win_condition: str, foul: bool = False):
        """End the game"""
        self.game_state = GameState.GAME_OVER
        
        if foul:
            # Current player loses due to foul
            winner = self.player2 if self.current_player == self.player1 else self.player1
        else:
            # Current player wins
            winner = self.current_player
            
        winner.score += 1
        print(f"Game Over! {winner.name} wins by {win_condition}")
        
    def call_pocket(self, pocket_name: str):
        """Call a pocket for eight ball shot"""
        if self.current_player.can_shoot_eight_ball():
            self.called_pocket = pocket_name
            print(f"{self.current_player.name} calls {pocket_name} for eight ball")
            
    def draw_game_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw game state overlay on frame"""
        overlay = frame.copy()
        
        # Create semi-transparent overlay
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Game information
        y = 30
        cv2.putText(frame, f"8-Ball Pool Game", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 30
        
        cv2.putText(frame, f"State: {self.game_state}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        
        cv2.putText(frame, f"Current: {self.current_player.name}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        
        cv2.putText(frame, f"Type: {self.current_player.player_type}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        
        cv2.putText(frame, f"Shot: {self.shot_count}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        
        if self.called_pocket:
            cv2.putText(frame, f"Called: {self.called_pocket}", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        return frame
        
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state as dictionary"""
        return {
            'game_state': self.game_state,
            'current_player': {
                'name': self.current_player.name,
                'type': self.current_player.player_type,
                'balls_remaining': self.current_player.balls_remaining,
                'can_shoot_eight': self.current_player.can_shoot_eight_ball()
            },
            'player1': {
                'name': self.player1.name,
                'type': self.player1.player_type,
                'score': self.player1.score,
                'balls_remaining': self.player1.balls_remaining
            },
            'player2': {
                'name': self.player2.name,
                'type': self.player2.player_type,
                'score': self.player2.score,
                'balls_remaining': self.player2.balls_remaining
            },
            'balls_on_table': list(self.balls_on_table),
            'cue_ball_on_table': self.cue_ball_on_table,
            'eight_ball_on_table': self.eight_ball_on_table,
            'shot_count': self.shot_count,
            'called_pocket': self.called_pocket
        }
        
    def save_game_log(self, filename: str):
        """Save game log to file"""
        game_log = {
            'game_state': self.get_game_state(),
            'shot_history': [
                {
                    'player': shot.player,
                    'shot_type': shot.shot_type,
                    'start_time': shot.start_time,
                    'end_time': shot.end_time,
                    'balls_pocketed': shot.balls_pocketed,
                    'fouls': shot.fouls,
                    'is_legal': shot.is_legal
                }
                for shot in self.shot_history
            ],
            'detection_history': self.detection_history[-100:]  # Last 100 detections
        }
        
        with open(filename, 'w') as f:
            json.dump(game_log, f, indent=2, default=str) 
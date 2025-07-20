import cv2
import numpy as np
import sys
import os
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.config import POCKET_DETECTION_RADIUS, POCKET_CONFIDENCE_THRESHOLD, POCKET_CLASSES

class PocketDetector:
    def __init__(self):
        """Initialize pocket detector for ball pocketing events"""
        self.pocket_events = []
        self.ball_tracking = {}  # Track ball positions over time
        self.pocket_positions = {}  # Store detected pocket positions by type
        self.frame_count = 0
        
    def update(self, detections, frame):
        """
        Update pocket detector with new detections
        
        Args:
            detections: list of detections [x1, y1, x2, y2, confidence, class_id]
            frame: current frame for visualization
            
        Returns:
            list of pocket events detected in this frame
        """
        self.frame_count += 1
        current_events = []
        
        # Extract balls and pockets from detections
        balls = [det for det in detections if det[5] == 0]  # class_id 0 = ball
        pockets = [det for det in detections if det[5] in POCKET_CLASSES]  # pocket classes
        
        # Update pocket positions
        self._update_pocket_positions(pockets)
        
        # Update ball tracking
        self._update_ball_tracking(balls)
        
        # Check for pocketing events
        pocketing_events = self._check_pocketing_events(balls, pockets)
        
        # Add new events to history
        for event in pocketing_events:
            event['frame'] = self.frame_count
            event['timestamp'] = time.time()
            self.pocket_events.append(event)
            current_events.append(event)
        
        return current_events
    
    def _update_pocket_positions(self, pockets):
        """Update stored pocket positions by type"""
        self.pocket_positions = {}
        for pocket in pockets:
            x1, y1, x2, y2, conf, class_id = pocket
            if conf >= POCKET_CONFIDENCE_THRESHOLD:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Get pocket type name
                pocket_type = f"class_{class_id}"  # Will be updated with actual names
                if pocket_type not in self.pocket_positions:
                    self.pocket_positions[pocket_type] = []
                self.pocket_positions[pocket_type].append((center_x, center_y))
    
    def _update_ball_tracking(self, balls):
        """Update ball position tracking"""
        current_balls = {}
        
        for ball in balls:
            x1, y1, x2, y2, conf, _ = ball
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ball_id = f"{center_x:.0f}_{center_y:.0f}"  # Simple ID based on position
            
            current_balls[ball_id] = {
                'position': (center_x, center_y),
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'last_seen': self.frame_count
            }
        
        # Update tracking history
        for ball_id, ball_info in current_balls.items():
            if ball_id in self.ball_tracking:
                # Update existing ball
                self.ball_tracking[ball_id].update(ball_info)
                self.ball_tracking[ball_id]['frames_tracked'] = self.ball_tracking[ball_id].get('frames_tracked', 0) + 1
            else:
                # New ball
                ball_info['frames_tracked'] = 1
                self.ball_tracking[ball_id] = ball_info
        
        # Remove old balls (not seen for more than 10 frames)
        old_balls = [ball_id for ball_id, info in self.ball_tracking.items() 
                    if self.frame_count - info['last_seen'] > 10]
        for ball_id in old_balls:
            del self.ball_tracking[ball_id]
    
    def _check_pocketing_events(self, balls, pockets):
        """Check for ball pocketing events"""
        events = []
        
        for ball in balls:
            x1, y1, x2, y2, conf, _ = ball
            ball_center_x = (x1 + x2) / 2
            ball_center_y = (y1 + y2) / 2
            
            # Check if ball is near any pocket
            for pocket in pockets:
                px1, py1, px2, py2, p_conf, p_class_id = pocket
                pocket_center_x = (px1 + px2) / 2
                pocket_center_y = (py1 + py2) / 2
                
                # Calculate distance between ball and pocket
                distance = np.sqrt((ball_center_x - pocket_center_x)**2 + 
                                 (ball_center_y - pocket_center_y)**2)
                
                # Check if ball is within pocket detection radius
                if distance <= POCKET_DETECTION_RADIUS:
                    # Check if this is a new pocketing event
                    ball_id = f"{ball_center_x:.0f}_{ball_center_y:.0f}"
                    
                    if self._is_new_pocketing_event(ball_id, pocket_center_x, pocket_center_y):
                        event = {
                            'ball_position': (ball_center_x, ball_center_y),
                            'pocket_position': (pocket_center_x, pocket_center_y),
                            'pocket_type': f"class_{p_class_id}",
                            'ball_confidence': conf,
                            'pocket_confidence': p_conf,
                            'distance': distance,
                            'ball_id': ball_id
                        }
                        events.append(event)
        
        return events
    
    def _is_new_pocketing_event(self, ball_id, pocket_x, pocket_y):
        """Check if this is a new pocketing event (not already recorded)"""
        # Check if this ball was recently pocketed
        for event in self.pocket_events[-10:]:  # Check last 10 events
            if (event.get('ball_id') == ball_id and 
                abs(event['pocket_position'][0] - pocket_x) < 10 and
                abs(event['pocket_position'][1] - pocket_y) < 10):
                return False
        
        return True
    
    def draw_pocket_events(self, frame, events):
        """
        Draw pocketing events on the frame
        
        Args:
            frame: numpy array of the input frame
            events: list of pocketing events
            
        Returns:
            frame with pocketing events drawn
        """
        for event in events:
            ball_pos = event['ball_position']
            pocket_pos = event['pocket_position']
            pocket_type = event.get('pocket_type', 'Unknown')
            
            # Draw line from ball to pocket
            cv2.line(frame, 
                    (int(ball_pos[0]), int(ball_pos[1])),
                    (int(pocket_pos[0]), int(pocket_pos[1])),
                    (0, 255, 255), 2)
            
            # Draw pocketing indicator
            cv2.putText(frame, f"POCKET! ({pocket_type})", 
                       (int(ball_pos[0]), int(ball_pos[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw pocket detection radius
            cv2.circle(frame, (int(pocket_pos[0]), int(pocket_pos[1])),
                      POCKET_DETECTION_RADIUS, (0, 255, 255), 2)
        
        return frame
    
    def get_pocket_statistics(self):
        """Get statistics about pocketing events"""
        if not self.pocket_events:
            return {}
        
        total_pockets = len(self.pocket_events)
        recent_pockets = len([e for e in self.pocket_events 
                            if time.time() - e['timestamp'] < 60])  # Last minute
        
        # Count by pocket type
        pocket_type_counts = {}
        for event in self.pocket_events:
            pocket_type = event.get('pocket_type', 'Unknown')
            pocket_type_counts[pocket_type] = pocket_type_counts.get(pocket_type, 0) + 1
        
        return {
            'total_pockets': total_pockets,
            'recent_pockets': recent_pockets,
            'pocket_rate': recent_pockets / 60.0 if recent_pockets > 0 else 0,  # pockets per second
            'pocket_type_counts': pocket_type_counts
        }
    
    def save_pocket_events(self, filename):
        """Save pocket events to file"""
        import json
        
        events_data = []
        for event in self.pocket_events:
            event_copy = event.copy()
            event_copy['timestamp'] = str(event_copy['timestamp'])  # Convert to string for JSON
            events_data.append(event_copy)
        
        with open(filename, 'w') as f:
            json.dump(events_data, f, indent=2) 
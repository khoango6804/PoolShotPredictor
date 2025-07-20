import cv2
import numpy as np
from supervision.tracker import DeepSORT
from ..config.config import TRACKING_BUFFER, MAX_DISAPPEARED

class BallTracker:
    def __init__(self):
        """Initialize the ball tracker with DeepSORT"""
        self.tracker = DeepSORT(
            max_age=MAX_DISAPPEARED,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=TRACKING_BUFFER
        )
        
    def update(self, detections, frame):
        """
        Update tracks with new detections
        
        Args:
            detections: list of detections in format [x1, y1, x2, y2, confidence, class_id]
            frame: numpy array of the input frame
            
        Returns:
            list of tracks in format [track_id, x1, y1, x2, y2]
        """
        if len(detections) == 0:
            return []
            
        # Convert detections to numpy array
        detections = np.array(detections)
        
        # Update tracks
        tracks = self.tracker.update(detections, frame)
        
        return tracks
    
    def draw_tracks(self, frame, tracks):
        """
        Draw tracking information on the frame
        
        Args:
            frame: numpy array of the input frame
            tracks: list of tracks
            
        Returns:
            frame with drawn tracks
        """
        for track in tracks:
            track_id, x1, y1, x2, y2 = track
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw track ID
            cv2.putText(frame, f"Track {track_id}", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
            
        return frame 
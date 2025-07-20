from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    MAX_DETECTIONS, MIN_BALL_SIZE, MAX_BALL_SIZE,
    CLASSES, CLASS_COLORS, MIN_TABLE_SIZE, MAX_TABLE_SIZE,
    MIN_POCKET_SIZE, MAX_POCKET_SIZE, POCKET_CLASSES
)

class MultiObjectDetector:
    def __init__(self, model_path=None):
        """Initialize the multi-object detector with YOLOv8 model"""
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(YOLO_MODEL)
        
    def detect(self, frame):
        """
        Detect multiple objects in the given frame
        
        Args:
            frame: numpy array of the input frame
            
        Returns:
            list of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        # Run detection
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        detections = []
        
        # Process detections
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = r
            class_id = int(class_id)
            
            # Calculate object size
            obj_width = x2 - x1
            obj_height = y2 - y1
            obj_size = max(obj_width, obj_height)
            
            # Filter by size based on class
            if self._is_valid_size(class_id, obj_size):
                detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Sort by confidence and limit number of detections
        detections.sort(key=lambda x: x[4], reverse=True)
        detections = detections[:MAX_DETECTIONS]
        
        return detections
    
    def _is_valid_size(self, class_id, obj_size):
        """Check if object size is valid for the given class"""
        if class_id == 0:  # ball
            return MIN_BALL_SIZE <= obj_size <= MAX_BALL_SIZE
        elif class_id == 1:  # table
            return MIN_TABLE_SIZE <= obj_size <= MAX_TABLE_SIZE
        elif class_id in POCKET_CLASSES:  # pocket classes
            return MIN_POCKET_SIZE <= obj_size <= MAX_POCKET_SIZE
        return True
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: numpy array of the input frame
            detections: list of detections
            
        Returns:
            frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # Get color for this class
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASSES.get(class_id, f"class_{class_id}")
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label with class name and confidence
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            
            # Draw center point for balls and pockets
            if class_id == 0 or class_id in POCKET_CLASSES:  # ball or pocket
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
        return frame
    
    def filter_overlapping_detections(self, detections):
        """
        Filter out overlapping detections keeping the one with highest confidence
        
        Args:
            detections: list of detections
            
        Returns:
            filtered detections
        """
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            overlap = False
            
            for f in filtered:
                fx1, fy1, fx2, fy2, _, f_class_id = f
                
                # Don't filter overlaps between different classes
                if class_id != f_class_id:
                    continue
                
                # Calculate IoU
                xx1 = max(x1, fx1)
                yy1 = max(y1, fy1)
                xx2 = min(x2, fx2)
                yy2 = min(y2, fy2)
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                intersection = w * h
                
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (fx2 - fx1) * (fy2 - fy1)
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > IOU_THRESHOLD:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(det)
                
        return filtered
    
    def get_balls(self, detections):
        """Extract only ball detections"""
        return [det for det in detections if det[5] == 0]
    
    def get_pockets(self, detections):
        """Extract only pocket detections"""
        return [det for det in detections if det[5] in POCKET_CLASSES]
    
    def get_tables(self, detections):
        """Extract only table detections"""
        return [det for det in detections if det[5] == 1]
    
    def get_all_pockets(self, detections):
        """Get all pocket detections with their specific types"""
        pockets = {}
        for det in detections:
            if det[5] in POCKET_CLASSES:
                pocket_type = CLASSES[det[5]]
                if pocket_type not in pockets:
                    pockets[pocket_type] = []
                pockets[pocket_type].append(det)
        return pockets 
"""
Improved Multi-Ball Detector
Phiên bản cải tiến để detect nhiều bóng cùng lúc tốt hơn
"""

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

class ImprovedMultiObjectDetector:
    def __init__(self, model_path=None, optimized_config=None):
        """
        Initialize the improved multi-object detector
        
        Args:
            model_path: Path to custom model
            optimized_config: Dictionary with optimized parameters
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(YOLO_MODEL)
        
        # Use optimized config if provided
        if optimized_config:
            self.confidence_threshold = optimized_config.get('confidence', CONFIDENCE_THRESHOLD)
            self.iou_threshold = optimized_config.get('iou', IOU_THRESHOLD)
            self.max_detections = optimized_config.get('max_detections', MAX_DETECTIONS)
            self.enable_size_filtering = optimized_config.get('enable_size_filtering', True)
            self.enable_overlap_filtering = optimized_config.get('enable_overlap_filtering', True)
        else:
            self.confidence_threshold = CONFIDENCE_THRESHOLD
            self.iou_threshold = IOU_THRESHOLD
            self.max_detections = MAX_DETECTIONS
            self.enable_size_filtering = True
            self.enable_overlap_filtering = True
        
        # Multi-scale detection settings
        self.multi_scale_enabled = True
        self.scale_factors = [0.8, 1.0, 1.2]  # Different scales to detect balls of various sizes
        
        print(f"🎯 Improved Detector initialized:")
        print(f"  Confidence: {self.confidence_threshold}")
        print(f"  IOU: {self.iou_threshold}")
        print(f"  Max Detections: {self.max_detections}")
        print(f"  Multi-scale: {self.multi_scale_enabled}")
        
    def detect(self, frame):
        """
        Detect multiple objects with improved multi-scale approach
        
        Args:
            frame: numpy array of the input frame
            
        Returns:
            list of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        all_detections = []
        
        if self.multi_scale_enabled:
            # Multi-scale detection
            for scale in self.scale_factors:
                # Resize frame
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                
                # Run detection on resized frame
                results = self.model(resized_frame, 
                                   conf=self.confidence_threshold, 
                                   iou=self.iou_threshold,
                                   verbose=False)[0]
                
                # Process detections and scale back coordinates
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = r
                    class_id = int(class_id)
                    
                    # Scale coordinates back to original size
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale
                    
                    # Calculate object size
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_size = max(obj_width, obj_height)
                    
                    # Apply size filtering if enabled
                    if not self.enable_size_filtering or self._is_valid_size(class_id, obj_size):
                        all_detections.append([x1, y1, x2, y2, confidence, class_id])
        else:
            # Single scale detection
            results = self.model(frame, 
                               conf=self.confidence_threshold, 
                               iou=self.iou_threshold,
                               verbose=False)[0]
            
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                class_id = int(class_id)
                
                # Calculate object size
                obj_width = x2 - x1
                obj_height = y2 - y1
                obj_size = max(obj_width, obj_height)
                
                # Apply size filtering if enabled
                if not self.enable_size_filtering or self._is_valid_size(class_id, obj_size):
                    all_detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Remove duplicate detections from multi-scale
        if self.multi_scale_enabled:
            all_detections = self._remove_duplicate_detections(all_detections)
        
        # Sort by confidence
        all_detections.sort(key=lambda x: x[4], reverse=True)
        
        # Apply overlap filtering if enabled
        if self.enable_overlap_filtering:
            all_detections = self.filter_overlapping_detections(all_detections)
        
        # Limit number of detections
        all_detections = all_detections[:self.max_detections]
        
        return all_detections
    
    def _remove_duplicate_detections(self, detections):
        """Remove duplicate detections from multi-scale approach"""
        if not detections:
            return []
        
        # Group detections by class
        class_groups = {}
        for det in detections:
            class_id = det[5]
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(det)
        
        # For each class, keep the detection with highest confidence
        filtered_detections = []
        for class_id, class_dets in class_groups.items():
            # Sort by confidence and keep the best one
            class_dets.sort(key=lambda x: x[4], reverse=True)
            filtered_detections.append(class_dets[0])
            
            # For balls, also keep nearby detections with good confidence
            if class_id == 0:  # ball class
                for det in class_dets[1:]:
                    # Check if this detection is far enough from the best one
                    best_det = class_dets[0]
                    distance = self._calculate_distance(det, best_det)
                    if distance > 50:  # 50 pixels threshold
                        filtered_detections.append(det)
        
        return filtered_detections
    
    def _calculate_distance(self, det1, det2):
        """Calculate distance between two detection centers"""
        x1_1, y1_1, x2_1, y2_1 = det1[:4]
        x1_2, y1_2, x2_2, y2_2 = det2[:4]
        
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def _is_valid_size(self, class_id, obj_size):
        """Check if object size is valid for the given class"""
        if class_id == 0:  # ball
            return MIN_BALL_SIZE <= obj_size <= MAX_BALL_SIZE
        elif class_id == 1:  # table
            return MIN_TABLE_SIZE <= obj_size <= MAX_TABLE_SIZE
        elif class_id in POCKET_CLASSES:  # pocket classes
            return MIN_POCKET_SIZE <= obj_size <= MAX_POCKET_SIZE
        return True
    
    def filter_overlapping_detections(self, detections):
        """
        Improved overlap filtering that's more lenient for balls
        
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
                
                # More lenient IOU threshold for balls
                threshold = self.iou_threshold * 0.7 if class_id == 0 else self.iou_threshold
                
                if iou > threshold:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(det)
                
        return filtered
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame with improved visualization
        
        Args:
            frame: numpy array of the input frame
            detections: list of detections
            
        Returns:
            frame with drawn detections
        """
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = det
            
            # Get color for this class
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASSES.get(class_id, f"class_{class_id}")
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(conf * 5))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label with class name and confidence
            label = f"{class_name}: {conf:.2f}"
            
            # Calculate text position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = int(x1)
            text_y = int(y1) - 10 if int(y1) - 10 > text_size[1] else int(y1) + text_size[1]
            
            # Draw text background
            cv2.rectangle(frame, 
                         (text_x, text_y - text_size[1]), 
                         (text_x + text_size[0], text_y + 5), 
                         color, -1)
            
            # Draw text
            cv2.putText(frame, label, 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
            
            # Draw center point for balls and pockets
            if class_id == 0 or class_id in POCKET_CLASSES:  # ball or pocket
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                
                # Add detection index for balls
                if class_id == 0:
                    cv2.putText(frame, str(i), 
                               (center_x + 5, center_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        return frame
    
    def get_balls(self, detections):
        """Extract only ball detections"""
        return [det for det in detections if det[5] == 0]
    
    def get_ball_count(self, detections):
        """Get count of detected balls"""
        return len(self.get_balls(detections))
    
    def get_detection_stats(self, detections):
        """Get statistics about detections"""
        balls = self.get_balls(detections)
        tables = [det for det in detections if det[5] == 1]
        pockets = [det for det in detections if det[5] in POCKET_CLASSES]
        
        return {
            'total_detections': len(detections),
            'ball_count': len(balls),
            'table_count': len(tables),
            'pocket_count': len(pockets),
            'avg_ball_confidence': np.mean([b[4] for b in balls]) if balls else 0,
            'ball_confidences': [b[4] for b in balls]
        } 
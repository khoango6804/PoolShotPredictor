"""
Multi-Model Detector for 8-Ball Pool
Sử dụng 3 model riêng biệt cho:
1. Table Detection
2. Ball Classification (bi số cụ thể)  
3. Pocket Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class MultiModelDetector:
    def __init__(self, 
                 table_model_path="models/table_model.pt",
                 ball_model_path="runs/detect/ball_classification/weights/best.pt", 
                 pocket_model_path="models/pocket_model.pt",
                 combined_model_path="models/billiards_model.pt"):
        """
        Initialize multi-model detector
        
        Args:
            table_model_path: Path to table detection model
            ball_model_path: Path to ball classification model  
            pocket_model_path: Path to pocket detection model
            combined_model_path: Fallback combined model
        """
        print("🎯 Initializing Multi-Model Detector...")
        
        # Load models with fallback
        self.table_model = self._load_model_safe(table_model_path, "Table")
        self.ball_model = self._load_model_safe(ball_model_path, "Ball Classification") 
        self.pocket_model = self._load_model_safe(pocket_model_path, "Pocket")
        self.combined_model = self._load_model_safe(combined_model_path, "Combined (Fallback)")
        
        # Ball class mapping for 8-ball pool
        self.ball_class_mapping = {
            0: "cue_ball",      # Bi trắng
            1: "solid_1", 2: "solid_2", 3: "solid_3", 4: "solid_4", 
            5: "solid_5", 6: "solid_6", 7: "solid_7",
            8: "eight_ball",    # Bi đen số 8
            9: "stripe_9", 10: "stripe_10", 11: "stripe_11", 12: "stripe_12",
            13: "stripe_13", 14: "stripe_14", 15: "stripe_15",
            94: "unknown_ball"  # Class đặc biệt
        }
        
        # Colors for visualization
        self.ball_colors = {
            "cue_ball": (255, 255, 255),      # Trắng
            "eight_ball": (0, 0, 0),          # Đen
            "solid_1": (255, 255, 0),         # Vàng cho solid
            "solid_2": (255, 255, 0), "solid_3": (255, 255, 0),
            "solid_4": (255, 255, 0), "solid_5": (255, 255, 0),
            "solid_6": (255, 255, 0), "solid_7": (255, 255, 0),
            "stripe_9": (255, 0, 255),        # Magenta cho stripe
            "stripe_10": (255, 0, 255), "stripe_11": (255, 0, 255),
            "stripe_12": (255, 0, 255), "stripe_13": (255, 0, 255),
            "stripe_14": (255, 0, 255), "stripe_15": (255, 0, 255),
            "table": (0, 255, 0),             # Xanh lá cho bàn
            "pocket": (255, 255, 0),          # Vàng cho lỗ
            "unknown_ball": (128, 128, 128)   # Xám cho bi không xác định
        }
        
    def _load_model_safe(self, model_path, model_name):
        """Safely load model with error handling"""
        try:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"✅ {model_name} model loaded: {model_path}")
                return model
            else:
                print(f"⚠️  {model_name} model not found: {model_path}")
                return None
        except Exception as e:
            print(f"❌ Error loading {model_name} model: {e}")
            return None
    
    def detect_all(self, frame, conf_threshold=0.5):
        """
        Run detection with all available models
        
        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold for detections
            
        Returns:
            dict: Combined detection results
        """
        results = {
            'tables': [],
            'balls': [],
            'pockets': [],
            'combined': []
        }
        
        # 1. Table Detection
        if self.table_model:
            try:
                table_results = self.table_model(frame, conf=conf_threshold, verbose=False)[0]
                for r in table_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = r
                    results['tables'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': 'table',
                        'class_id': int(class_id)
                    })
            except Exception as e:
                print(f"⚠️ Table detection error: {e}")
        
        # 2. Ball Classification  
        if self.ball_model:
            try:
                ball_results = self.ball_model(frame, conf=conf_threshold, verbose=False)[0]
                for r in ball_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = r
                    class_id = int(class_id)
                    ball_type = self.ball_class_mapping.get(class_id, f"ball_{class_id}")
                    
                    results['balls'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': ball_type,
                        'class_id': class_id,
                        'ball_number': class_id if class_id <= 15 else None
                    })
            except Exception as e:
                print(f"⚠️ Ball detection error: {e}")
        
        # 3. Pocket Detection
        if self.pocket_model:
            try:
                pocket_results = self.pocket_model(frame, conf=conf_threshold, verbose=False)[0]
                for r in pocket_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = r
                    results['pockets'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': f"pocket_{int(class_id)}",
                        'class_id': int(class_id)
                    })
            except Exception as e:
                print(f"⚠️ Pocket detection error: {e}")
        
        # 4. Fallback Combined Model
        if self.combined_model and (not results['balls'] or not results['tables']):
            try:
                combined_results = self.combined_model(frame, conf=conf_threshold, verbose=False)[0]
                for r in combined_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = r
                    results['combined'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': int(class_id)
                    })
            except Exception as e:
                print(f"⚠️ Combined detection error: {e}")
        
        return results
    
    def draw_detections(self, frame, results):
        """Draw all detections on frame"""
        annotated_frame = frame.copy()
        
        # Draw tables
        for det in results['tables']:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            color = self.ball_colors.get('table', (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated_frame, f"Table: {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw balls with specific types
        for det in results['balls']:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            ball_type = det['class']
            color = self.ball_colors.get(ball_type, (255, 255, 255))
            
            # Thicker border for balls
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Ball number in center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Label with ball number
            if det.get('ball_number') is not None:
                label = f"Ball {det['ball_number']}: {det['confidence']:.2f}"
            else:
                label = f"{ball_type}: {det['confidence']:.2f}"
            
            cv2.putText(annotated_frame, label,
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw pockets
        for det in results['pockets']:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            color = self.ball_colors.get('pocket', (255, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"Pocket: {det['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame
    
    def get_ball_summary(self, results):
        """Get summary of detected balls for game logic"""
        summary = {
            'cue_ball': None,
            'eight_ball': None,
            'solids': [],      # Bi 1-7
            'stripes': [],     # Bi 9-15
            'total_balls': len(results['balls'])
        }
        
        for ball in results['balls']:
            ball_type = ball['class']
            ball_num = ball.get('ball_number')
            
            if ball_type == 'cue_ball':
                summary['cue_ball'] = ball
            elif ball_type == 'eight_ball':
                summary['eight_ball'] = ball
            elif ball_type.startswith('solid_') and ball_num and 1 <= ball_num <= 7:
                summary['solids'].append(ball)
            elif ball_type.startswith('stripe_') and ball_num and 9 <= ball_num <= 15:
                summary['stripes'].append(ball)
        
        return summary

# Demo function
def main():
    """Demo the multi-model detector"""
    detector = MultiModelDetector()
    
    # Test with a sample image or camera
    cap = cv2.VideoCapture(0)  # Use camera
    
    print("🎱 Multi-Model Detection Started!")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = detector.detect_all(frame)
        
        # Draw results
        annotated_frame = detector.draw_detections(frame, results)
        
        # Get ball summary
        ball_summary = detector.get_ball_summary(results)
        
        # Display info
        info_text = f"Balls: {ball_summary['total_balls']} | "
        info_text += f"Solids: {len(ball_summary['solids'])} | "
        info_text += f"Stripes: {len(ball_summary['stripes'])}"
        
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Multi-Model 8-Ball Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
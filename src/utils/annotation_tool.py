import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from collections import defaultdict

class BilliardsAnnotationTool:
    def __init__(self, images_dir, labels_dir):
        """
        Initialize annotation tool for billiards multi-class detection
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory to save annotations
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Class definitions
        self.classes = {
            'ball': 0,
            'table_edge': 1,
            'cue_stick': 2,
            'pocket': 3
        }
        
        self.class_colors = {
            'ball': (0, 255, 0),      # Green
            'table_edge': (255, 0, 0), # Blue
            'cue_stick': (0, 0, 255),  # Red
            'pocket': (255, 255, 0)    # Cyan
        }
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
        
        self.image_files.sort()
        self.current_image_idx = 0
        
        # Annotation state
        self.current_class = 'ball'
        self.annotations = defaultdict(list)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Window setup
        self.window_name = 'Billiards Annotation Tool'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                self.add_annotation()
    
    def add_annotation(self):
        """Add annotation for the drawn bounding box"""
        if not self.start_point or not self.end_point:
            return
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        annotation = {
            'position': [center_x, center_y],
            'size': [width, height]
        }
        
        self.annotations[self.current_class].append(annotation)
        print(f"Added {self.current_class} annotation: center=({center_x:.1f}, {center_y:.1f}), size=({width:.1f}, {height:.1f})")
    
    def load_annotations(self, image_path):
        """Load existing annotations for an image"""
        label_path = self.labels_dir / f"{image_path.stem}.json"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                data = json.load(f)
                self.annotations.clear()
                
                # Convert to our format
                for class_name in self.classes.keys():
                    if class_name in data:
                        self.annotations[class_name] = data[class_name]
        else:
            self.annotations.clear()
    
    def save_annotations(self, image_path):
        """Save annotations to JSON file"""
        if not self.annotations:
            return
        
        label_path = self.labels_dir / f"{image_path.stem}.json"
        
        # Read image to get dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        height, width = img.shape[:2]
        
        # Prepare data
        data = {
            'image_width': width,
            'image_height': height,
            'balls': self.annotations['ball'],
            'table_edges': self.annotations['table_edge'],
            'cue_sticks': self.annotations['cue_stick'],
            'pockets': self.annotations['pocket']
        }
        
        with open(label_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Annotations saved to: {label_path}")
    
    def draw_annotations(self, img):
        """Draw all annotations on the image"""
        img_copy = img.copy()
        
        # Draw existing annotations
        for class_name, annotations in self.annotations.items():
            color = self.class_colors[class_name]
            for ann in annotations:
                x, y = ann['position']
                w, h = ann['size']
                
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_copy, class_name, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current bounding box being drawn
        if self.drawing and self.start_point and self.end_point:
            color = self.class_colors[self.current_class]
            cv2.rectangle(img_copy, self.start_point, self.end_point, color, 2)
        
        return img_copy
    
    def show_help(self, img):
        """Show help text on the image"""
        help_text = [
            "Controls:",
            "1-4: Switch class (1=ball, 2=table_edge, 3=cue_stick, 4=pocket)",
            "Mouse: Draw bounding boxes",
            "S: Save annotations",
            "N: Next image",
            "P: Previous image",
            "D: Delete last annotation",
            "C: Clear all annotations",
            "Q: Quit"
        ]
        
        y_offset = 30
        for text in help_text:
            cv2.putText(img, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Show current class
        cv2.putText(img, f"Current class: {self.current_class}", (10, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.class_colors[self.current_class], 3)
        
        return img
    
    def run(self):
        """Run the annotation tool"""
        if not self.image_files:
            print("No images found!")
            return
        
        print(f"Found {len(self.image_files)} images")
        print("Starting annotation tool...")
        
        while self.current_image_idx < len(self.image_files):
            image_path = self.image_files[self.current_image_idx]
            
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Could not load image: {image_path}")
                self.current_image_idx += 1
                continue
            
            # Load existing annotations
            self.load_annotations(image_path)
            
            print(f"\nAnnotating image {self.current_image_idx + 1}/{len(self.image_files)}: {image_path.name}")
            print(f"Current class: {self.current_class}")
            
            while True:
                # Draw annotations and help
                display_img = self.draw_annotations(img)
                display_img = self.show_help(display_img)
                
                # Show image info
                cv2.putText(display_img, f"Image: {image_path.name}", 
                           (10, display_img.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(self.window_name, display_img)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Save and quit
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):
                    # Save annotations
                    self.save_annotations(image_path)
                elif key == ord('n'):
                    # Next image
                    self.save_annotations(image_path)
                    self.current_image_idx += 1
                    break
                elif key == ord('p'):
                    # Previous image
                    self.save_annotations(image_path)
                    self.current_image_idx = max(0, self.current_image_idx - 1)
                    break
                elif key == ord('d'):
                    # Delete last annotation
                    if self.annotations[self.current_class]:
                        self.annotations[self.current_class].pop()
                        print(f"Deleted last {self.current_class} annotation")
                elif key == ord('c'):
                    # Clear all annotations
                    self.annotations.clear()
                    print("Cleared all annotations")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    # Switch class
                    class_map = {'1': 'ball', '2': 'table_edge', '3': 'cue_stick', '4': 'pocket'}
                    self.current_class = class_map[chr(key)]
                    print(f"Switched to class: {self.current_class}")
        
        print("Annotation completed!")

def main():
    parser = argparse.ArgumentParser(description='Billiards Multi-Class Annotation Tool')
    parser.add_argument('--images', '-i', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--labels', '-l', type=str, required=True,
                       help='Directory to save annotations')
    
    args = parser.parse_args()
    
    # Create annotation tool
    tool = BilliardsAnnotationTool(args.images, args.labels)
    
    # Run the tool
    tool.run()

if __name__ == "__main__":
    main() 
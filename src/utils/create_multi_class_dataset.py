import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm

class MultiClassDatasetCreator:
    def __init__(self, output_dir):
        """
        Initialize dataset creator for multi-class billiards detection
        
        Args:
            output_dir: Directory to save the dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class definitions
        self.classes = {
            'ball': 0,
            'table_edge': 1,
            'cue_stick': 2,
            'pocket': 3
        }
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        
    def create_synthetic_dataset(self, num_images=100):
        """
        Create synthetic dataset with all classes
        
        Args:
            num_images: Number of synthetic images to create
        """
        print(f"Creating {num_images} synthetic images...")
        
        for i in tqdm(range(num_images)):
            # Create synthetic image
            img = self._create_synthetic_image()
            
            # Create annotations
            annotations = self._create_synthetic_annotations(img.shape)
            
            # Save image and annotations
            img_path = self.output_dir / 'images' / f'synthetic_{i:04d}.jpg'
            label_path = self.output_dir / 'labels' / f'synthetic_{i:04d}.json'
            
            cv2.imwrite(str(img_path), img)
            self._save_annotations(annotations, label_path, img.shape)
        
        print(f"Synthetic dataset created in: {self.output_dir}")
    
    def _create_synthetic_image(self):
        """Create a synthetic billiards table image"""
        # Create green table background
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:, :] = (34, 139, 34)  # Forest green
        
        # Add table border (darker green)
        cv2.rectangle(img, (50, 50), (1870, 1030), (0, 100, 0), 100)
        
        # Add some texture/noise
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def _create_synthetic_annotations(self, img_shape):
        """Create synthetic annotations for all classes"""
        height, width = img_shape[:2]
        annotations = {
            'image_width': width,
            'image_height': height,
            'balls': [],
            'table_edges': [],
            'cue_sticks': [],
            'pockets': []
        }
        
        # Add table edges (4 corners)
        edge_width, edge_height = 200, 50
        edges = [
            (100, 100, edge_width, edge_height),  # Top-left
            (width - 300, 100, edge_width, edge_height),  # Top-right
            (100, height - 150, edge_width, edge_height),  # Bottom-left
            (width - 300, height - 150, edge_width, edge_height)  # Bottom-right
        ]
        
        for x, y, w, h in edges:
            annotations['table_edges'].append({
                'position': [x + w//2, y + h//2],
                'size': [w, h]
            })
        
        # Add pockets (6 corners)
        pocket_size = 40
        pockets = [
            (100, 100),  # Top-left
            (width//2, 80),  # Top-center
            (width - 140, 100),  # Top-right
            (100, height - 140),  # Bottom-left
            (width//2, height - 120),  # Bottom-center
            (width - 140, height - 140)  # Bottom-right
        ]
        
        for x, y in pockets:
            annotations['pockets'].append({
                'position': [x, y],
                'size': [pocket_size, pocket_size]
            })
        
        # Add balls (random positions)
        num_balls = np.random.randint(5, 15)
        for _ in range(num_balls):
            x = np.random.randint(200, width - 200)
            y = np.random.randint(200, height - 200)
            ball_size = np.random.randint(20, 40)
            
            annotations['balls'].append({
                'position': [x, y],
                'size': [ball_size, ball_size]
            })
        
        # Add cue stick (random position and angle)
        if np.random.random() > 0.3:  # 70% chance to have cue stick
            x = np.random.randint(300, width - 300)
            y = np.random.randint(300, height - 300)
            length = np.random.randint(100, 200)
            width_stick = np.random.randint(8, 15)
            
            annotations['cue_sticks'].append({
                'position': [x, y],
                'size': [length, width_stick]
            })
        
        return annotations
    
    def _save_annotations(self, annotations, label_path, img_shape):
        """Save annotations to JSON file"""
        with open(label_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def convert_existing_dataset(self, existing_dataset_path):
        """
        Convert existing ball-only dataset to multi-class format
        
        Args:
            existing_dataset_path: Path to existing dataset with only balls
        """
        existing_path = Path(existing_dataset_path)
        
        if not existing_path.exists():
            print(f"Error: Dataset path not found: {existing_path}")
            return
        
        print(f"Converting existing dataset: {existing_path}")
        
        # Find all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(existing_path.rglob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images to convert")
        
        for img_path in tqdm(image_files):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Read existing ball annotations
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                continue
            
            # Convert YOLO format to our JSON format
            annotations = self._convert_yolo_to_json(label_path, img.shape)
            
            # Add synthetic table edges and pockets
            annotations = self._add_table_elements(annotations, img.shape)
            
            # Save converted annotations
            new_label_path = self.output_dir / 'labels' / f"{img_path.stem}.json"
            self._save_annotations(annotations, new_label_path, img.shape)
            
            # Copy image
            new_img_path = self.output_dir / 'images' / img_path.name
            cv2.imwrite(str(new_img_path), img)
        
        print(f"Dataset conversion completed. Output: {self.output_dir}")
    
    def _convert_yolo_to_json(self, yolo_path, img_shape):
        """Convert YOLO format to JSON format"""
        height, width = img_shape[:2]
        
        annotations = {
            'image_width': width,
            'image_height': height,
            'balls': [],
            'table_edges': [],
            'cue_sticks': [],
            'pockets': []
        }
        
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    if class_id == 0:  # ball
                        annotations['balls'].append({
                            'position': [x_center, y_center],
                            'size': [w, h]
                        })
        
        return annotations
    
    def _add_table_elements(self, annotations, img_shape):
        """Add synthetic table edges and pockets to existing annotations"""
        height, width = img_shape[:2]
        
        # Add table edges if not present
        if not annotations['table_edges']:
            edge_width, edge_height = 200, 50
            edges = [
                (100, 100, edge_width, edge_height),
                (width - 300, 100, edge_width, edge_height),
                (100, height - 150, edge_width, edge_height),
                (width - 300, height - 150, edge_width, edge_height)
            ]
            
            for x, y, w, h in edges:
                annotations['table_edges'].append({
                    'position': [x + w//2, y + h//2],
                    'size': [w, h]
                })
        
        # Add pockets if not present
        if not annotations['pockets']:
            pocket_size = 40
            pockets = [
                (100, 100),
                (width//2, 80),
                (width - 140, 100),
                (100, height - 140),
                (width//2, height - 120),
                (width - 140, height - 140)
            ]
            
            for x, y in pockets:
                annotations['pockets'].append({
                    'position': [x, y],
                    'size': [pocket_size, pocket_size]
                })
        
        return annotations
    
    def create_dataset_info(self):
        """Create dataset information file"""
        # Count files
        num_images = len(list((self.output_dir / 'images').glob('*.jpg')))
        num_labels = len(list((self.output_dir / 'labels').glob('*.json')))
        
        info = {
            'total_images': num_images,
            'total_labels': num_labels,
            'classes': list(self.classes.keys()),
            'class_mapping': self.classes,
            'image_size': (1920, 1080)
        }
        
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Dataset info saved to: {info_path}")
        return info

def main():
    parser = argparse.ArgumentParser(description='Create Multi-Class Billiards Dataset')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--synthetic', '-s', type=int, default=0,
                       help='Number of synthetic images to create')
    parser.add_argument('--convert', '-c', type=str,
                       help='Convert existing dataset (path to existing dataset)')
    
    args = parser.parse_args()
    
    # Create dataset creator
    creator = MultiClassDatasetCreator(args.output)
    
    if args.synthetic > 0:
        # Create synthetic dataset
        creator.create_synthetic_dataset(args.synthetic)
    
    if args.convert:
        # Convert existing dataset
        creator.convert_existing_dataset(args.convert)
    
    # Create dataset info
    creator.create_dataset_info()
    
    print(f"\nDataset created successfully in: {args.output}")
    print("Next steps:")
    print("1. Review and manually annotate if needed")
    print("2. Use dataset_processor.py to split into train/val/test")
    print("3. Use train_multi_class.py to train the model")

if __name__ == "__main__":
    main() 
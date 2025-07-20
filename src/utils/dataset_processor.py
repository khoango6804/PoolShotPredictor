import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json

class DatasetProcessor:
    def __init__(self, dataset_path, output_path):
        """
        Initialize dataset processor
        
        Args:
            dataset_path: Path to raw dataset
            output_path: Path to save processed dataset
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_path = self.output_path / "training"
        self.val_path = self.output_path / "validation"
        self.test_path = self.output_path / "testing"
        
        # Class definitions
        self.classes = {
            'ball': 0,
            'table_edge': 1,
            'cue_stick': 2,
            'pocket': 3
        }
        
        # Create output directories
        for path in [self.train_path, self.val_path, self.test_path]:
            path.mkdir(parents=True, exist_ok=True)
            (path / "images").mkdir(exist_ok=True)
            (path / "labels").mkdir(exist_ok=True)
    
    def process_dataset(self, split_ratio=(0.7, 0.2, 0.1)):
        """
        Process the dataset and split into train/val/test
        
        Args:
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
        """
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(self.dataset_path.rglob(f'*{ext}')))
        
        # Shuffle files
        np.random.shuffle(image_files)
        
        # Split files
        n_files = len(image_files)
        n_train = int(n_files * split_ratio[0])
        n_val = int(n_files * split_ratio[1])
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Process each split
        self._process_split(train_files, self.train_path, "train")
        self._process_split(val_files, self.val_path, "val")
        self._process_split(test_files, self.test_path, "test")
        
        # Create dataset info
        self._create_dataset_info()
    
    def _process_split(self, files, output_path, split_name):
        """
        Process a split of the dataset
        
        Args:
            files: List of image files
            output_path: Path to save processed files
            split_name: Name of the split (train/val/test)
        """
        print(f"Processing {split_name} split...")
        for img_path in tqdm(files):
            # Process image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Save processed image
            img_name = img_path.stem
            cv2.imwrite(str(output_path / "images" / f"{img_name}.jpg"), img)
            
            # Process and save labels if they exist
            label_path = img_path.with_suffix('.json')
            if label_path.exists():
                self._process_label(label_path, output_path / "labels" / f"{img_name}.txt", img.shape)
    
    def _process_label(self, label_path, output_path, img_shape):
        """
        Process label file and convert to YOLO format
        
        Args:
            label_path: Path to original label file
            output_path: Path to save processed label
            img_shape: Image shape (height, width, channels)
        """
        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            
            # Convert to YOLO format
            # Format: class_id x_center y_center width height
            # All values normalized to [0, 1]
            yolo_labels = []
            
            img_height, img_width = img_shape[:2]
            
            # Process balls
            for ball in label_data.get('balls', []):
                x, y = ball['position']
                w, h = ball['size']
                
                # Normalize coordinates
                x_center = x / img_width
                y_center = y / img_height
                width = w / img_width
                height = h / img_height
                
                # Add to labels
                yolo_labels.append(f"{self.classes['ball']} {x_center} {y_center} {width} {height}")
            
            # Process table edges
            for edge in label_data.get('table_edges', []):
                x, y = edge['position']
                w, h = edge['size']
                
                # Normalize coordinates
                x_center = x / img_width
                y_center = y / img_height
                width = w / img_width
                height = h / img_height
                
                # Add to labels
                yolo_labels.append(f"{self.classes['table_edge']} {x_center} {y_center} {width} {height}")
            
            # Process cue sticks
            for cue in label_data.get('cue_sticks', []):
                x, y = cue['position']
                w, h = cue['size']
                
                # Normalize coordinates
                x_center = x / img_width
                y_center = y / img_height
                width = w / img_width
                height = h / img_height
                
                # Add to labels
                yolo_labels.append(f"{self.classes['cue_stick']} {x_center} {y_center} {width} {height}")
            
            # Process pockets
            for pocket in label_data.get('pockets', []):
                x, y = pocket['position']
                w, h = pocket['size']
                
                # Normalize coordinates
                x_center = x / img_width
                y_center = y / img_height
                width = w / img_width
                height = h / img_height
                
                # Add to labels
                yolo_labels.append(f"{self.classes['pocket']} {x_center} {y_center} {width} {height}")
            
            # Save labels
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
                
        except Exception as e:
            print(f"Error processing label {label_path}: {e}")
    
    def _create_dataset_info(self):
        """Create dataset information file"""
        info = {
            'train': len(list(self.train_path.glob('images/*.jpg'))),
            'val': len(list(self.val_path.glob('images/*.jpg'))),
            'test': len(list(self.test_path.glob('images/*.jpg'))),
            'classes': list(self.classes.keys()),
            'class_mapping': self.classes,
            'image_size': (1920, 1080)
        }
        
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=4)
    
    def create_yaml_config(self, output_path=None):
        """
        Create YAML configuration file for YOLO training
        
        Args:
            output_path: Path to save YAML file (default: dataset output path)
        """
        if output_path is None:
            output_path = self.output_path
        
        yaml_content = f"""# YOLO dataset configuration
path: {self.output_path.absolute()}  # dataset root dir
train: training/images  # train images (relative to 'path')
val: validation/images  # val images (relative to 'path')
test: testing/images  # test images (relative to 'path')

# Classes
nc: {len(self.classes)}  # number of classes
names: {list(self.classes.keys())}  # class names
"""
        
        yaml_path = output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"YAML configuration saved to: {yaml_path}")
        return yaml_path 
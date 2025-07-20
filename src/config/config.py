import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
YOLO_MODEL = "yolov8m.pt"  # Using YOLOv8 medium for better detection
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to detect more objects
IOU_THRESHOLD = 0.3  # Lower IOU threshold to allow more detections
MAX_DETECTIONS = 30  # Maximum number of objects to detect

# Class definitions for real datasets
CLASSES = {
    # Ball detection (from existing dataset)
    0: "ball",
    
    # Table detection (from table detector dataset)
    1: "table",
    
    # Pocket detection (from pocket detection dataset)
    2: "BottomLeft",
    3: "BottomRight", 
    4: "IntersectionLeft",
    5: "IntersectionRight",
    6: "MediumLeft",
    7: "MediumRight",
    8: "SemicircleLeft",
    9: "SemicircleRight",
    10: "TopLeft",
    11: "TopRight"
}

# Pocket class mapping for easier access
POCKET_CLASSES = {
    2: "BottomLeft",
    3: "BottomRight", 
    4: "IntersectionLeft",
    5: "IntersectionRight",
    6: "MediumLeft",
    7: "MediumRight",
    8: "SemicircleLeft",
    9: "SemicircleRight",
    10: "TopLeft",
    11: "TopRight"
}

# Colors for visualization
CLASS_COLORS = {
    0: (0, 255, 0),      # Green for balls
    1: (255, 0, 0),      # Blue for table
    2: (255, 255, 0),    # Cyan for BottomLeft
    3: (255, 255, 0),    # Cyan for BottomRight
    4: (255, 255, 0),    # Cyan for IntersectionLeft
    5: (255, 255, 0),    # Cyan for IntersectionRight
    6: (255, 255, 0),    # Cyan for MediumLeft
    7: (255, 255, 0),    # Cyan for MediumRight
    8: (255, 255, 0),    # Cyan for SemicircleLeft
    9: (255, 255, 0),    # Cyan for SemicircleRight
    10: (255, 255, 0),   # Cyan for TopLeft
    11: (255, 255, 0)    # Cyan for TopRight
}

# Tracking configurations
TRACKING_BUFFER = 30  # Number of frames to keep in tracking buffer
MAX_DISAPPEARED = 30  # Maximum number of frames an object can be missing

# Camera configurations
CAMERA_WIDTH = 1920  # Higher resolution for better detection
CAMERA_HEIGHT = 1080
FPS = 30

# Detection specific configurations
MIN_BALL_SIZE = 20  # Minimum size of ball in pixels
MAX_BALL_SIZE = 100  # Maximum size of ball in pixels
MIN_TABLE_SIZE = 200  # Minimum size of table in pixels
MAX_TABLE_SIZE = 1000  # Maximum size of table in pixels
MIN_POCKET_SIZE = 30  # Minimum size of pocket in pixels
MAX_POCKET_SIZE = 80  # Maximum size of pocket in pixels

# Pocket detection settings
POCKET_DETECTION_RADIUS = 50  # Radius around pocket to detect ball entry
POCKET_CONFIDENCE_THRESHOLD = 0.5  # Higher confidence for pocket detection

# Dataset paths
POCKET_DATASET_PATH = PROJECT_ROOT / "pocket detection"
TABLE_DATASET_PATH = PROJECT_ROOT / "table detector"
BILLIARDS_DATASET_PATH = PROJECT_ROOT / "billiards-2"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 
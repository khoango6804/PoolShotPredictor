"""
Optimized Configuration for Multi-Ball Detection
Cấu hình tối ưu để detect nhiều bóng cùng lúc
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================================
# OPTIMIZED MODEL CONFIGURATIONS FOR MULTI-BALL DETECTION
# ============================================================================

# Model configurations - Optimized for detecting multiple balls
YOLO_MODEL = "yolov8m.pt"  # Using YOLOv8 medium for better detection
CONFIDENCE_THRESHOLD = 0.15  # Lowered from 0.3 to detect more balls
IOU_THRESHOLD = 0.2  # Lowered from 0.3 to allow overlapping balls
MAX_DETECTIONS = 100  # Increased from 30 to detect more objects

# ============================================================================
# RELAXED SIZE FILTERS FOR BETTER BALL DETECTION
# ============================================================================

# Size filters - Relaxed for better multi-ball detection
MIN_BALL_SIZE = 15  # Reduced from 20 to detect smaller balls
MAX_BALL_SIZE = 150  # Increased from 100 to detect larger balls
MIN_TABLE_SIZE = 200  # Minimum size of table in pixels
MAX_TABLE_SIZE = 1000  # Maximum size of table in pixels
MIN_POCKET_SIZE = 30  # Minimum size of pocket in pixels
MAX_POCKET_SIZE = 80  # Maximum size of pocket in pixels

# ============================================================================
# MULTI-SCALE DETECTION SETTINGS
# ============================================================================

# Enable multi-scale detection for better ball detection at different sizes
MULTI_SCALE_ENABLED = True
SCALE_FACTORS = [0.8, 1.0, 1.2]  # Different scales to detect balls of various sizes

# ============================================================================
# OVERLAP FILTERING SETTINGS
# ============================================================================

# Overlap filtering - More lenient for balls
ENABLE_OVERLAP_FILTERING = True
BALL_IOU_THRESHOLD_MULTIPLIER = 0.7  # More lenient IOU threshold for balls
MIN_BALL_DISTANCE = 30  # Minimum distance between ball centers (pixels)

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

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

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Colors for visualization (BGR format)
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

# ============================================================================
# TRACKING CONFIGURATIONS
# ============================================================================

# Tracking configurations
TRACKING_BUFFER = 30  # Number of frames to keep in tracking buffer
MAX_DISAPPEARED = 30  # Maximum number of frames an object can be missing

# ============================================================================
# CAMERA CONFIGURATIONS
# ============================================================================

# Camera configurations
CAMERA_WIDTH = 1920  # Higher resolution for better detection
CAMERA_HEIGHT = 1080
FPS = 30

# ============================================================================
# POCKET DETECTION SETTINGS
# ============================================================================

# Pocket detection settings
POCKET_DETECTION_RADIUS = 50  # Radius around pocket to detect ball entry
POCKET_CONFIDENCE_THRESHOLD = 0.5  # Higher confidence for pocket detection

# ============================================================================
# DATASET PATHS
# ============================================================================

# Dataset paths
POCKET_DATASET_PATH = PROJECT_ROOT / "pocket detection"
TABLE_DATASET_PATH = PROJECT_ROOT / "table detector"
BILLIARDS_DATASET_PATH = PROJECT_ROOT / "billiards-2"

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

# Performance settings
BATCH_SIZE = 1
DEVICE = "auto"  # Use GPU if available
ENABLE_SIZE_FILTERING = False  # Disable size filtering for better detection
ENABLE_OVERLAP_FILTERING = True  # Keep overlap filtering but with relaxed settings

# ============================================================================
# DEBUG AND LOGGING
# ============================================================================

# Debug settings
DEBUG_MODE = False
SAVE_DETECTION_IMAGES = True
LOG_DETECTION_STATS = True

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

# Preset configurations for different scenarios
CONFIG_PRESETS = {
    "default": {
        "confidence": 0.3,
        "iou": 0.3,
        "max_detections": 30,
        "enable_size_filtering": True,
        "enable_overlap_filtering": True
    },
    "multi_ball": {
        "confidence": 0.15,
        "iou": 0.2,
        "max_detections": 100,
        "enable_size_filtering": False,
        "enable_overlap_filtering": False
    },
    "high_sensitivity": {
        "confidence": 0.1,
        "iou": 0.1,
        "max_detections": 150,
        "enable_size_filtering": False,
        "enable_overlap_filtering": False
    },
    "balanced": {
        "confidence": 0.2,
        "iou": 0.25,
        "max_detections": 80,
        "enable_size_filtering": False,
        "enable_overlap_filtering": True
    }
}

def get_preset_config(preset_name="multi_ball"):
    """Get configuration preset"""
    return CONFIG_PRESETS.get(preset_name, CONFIG_PRESETS["default"])

def get_optimized_config():
    """Get the current optimized configuration"""
    return {
        "confidence": CONFIDENCE_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "enable_size_filtering": ENABLE_SIZE_FILTERING,
        "enable_overlap_filtering": ENABLE_OVERLAP_FILTERING,
        "multi_scale_enabled": MULTI_SCALE_ENABLED,
        "scale_factors": SCALE_FACTORS,
        "ball_iou_threshold_multiplier": BALL_IOU_THRESHOLD_MULTIPLIER,
        "min_ball_distance": MIN_BALL_DISTANCE
    } 
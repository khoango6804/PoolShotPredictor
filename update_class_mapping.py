#!/usr/bin/env python3
"""
Update class mapping for new dataset with 23 classes
"""

# New class mapping for 23 classes
NEW_BALL_CLASS_MAPPING = {
    0: "bag1",
    1: "bag2", 
    2: "bag3",
    3: "bag4",
    4: "bag5",
    5: "bag6",
    6: "ball0",      # cue ball
    7: "ball1",      # solid 1
    8: "ball10",     # stripe 10
    9: "ball11",     # stripe 11
    10: "ball12",    # stripe 12
    11: "ball13",    # stripe 13
    12: "ball14",    # stripe 14
    13: "ball15",    # stripe 15
    14: "ball2",     # solid 2
    15: "ball3",     # solid 3
    16: "ball4",     # solid 4
    17: "ball5",     # solid 5
    18: "ball6",     # solid 6
    19: "ball7",     # solid 7
    20: "ball8",     # eight ball
    21: "ball9",     # stripe 9
    22: "flag"
}

# Colors for different ball types
NEW_BALL_COLORS = {
    0: (255, 0, 0),      # Red for bag1
    1: (0, 255, 0),      # Green for bag2
    2: (0, 0, 255),      # Blue for bag3
    3: (255, 255, 0),    # Yellow for bag4
    4: (255, 0, 255),    # Magenta for bag5
    5: (0, 255, 255),    # Cyan for bag6
    6: (255, 255, 255),  # White for ball0 (cue ball)
    7: (255, 128, 0),    # Orange for ball1
    8: (128, 255, 0),    # Light green for ball10
    9: (0, 128, 255),    # Light blue for ball11
    10: (255, 0, 128),   # Pink for ball12
    11: (128, 0, 255),   # Purple for ball13
    12: (0, 255, 128),   # Light cyan for ball14
    13: (255, 128, 128), # Light red for ball15
    14: (128, 128, 0),   # Olive for ball2
    15: (0, 128, 128),   # Teal for ball3
    16: (128, 0, 128),   # Maroon for ball4
    17: (255, 165, 0),   # Orange for ball5
    18: (0, 255, 0),     # Green for ball6
    19: (128, 0, 0),     # Dark red for ball7
    20: (0, 0, 0),       # Black for ball8 (eight ball)
    21: (255, 215, 0),   # Gold for ball9
    22: (255, 255, 255)  # White for flag
}

def update_video_detector():
    """Update the video detector with new class mapping"""
    
    print("🔄 Updating video detector with new class mapping...")
    
    # Read current video detector
    with open('src/detectors/multi_class_video_detector.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace class mapping
    old_mapping_start = "# Ball class mapping for 8-ball pool (matching dataset)"
    old_mapping_end = "}"
    
    new_mapping = '''# Ball class mapping for new dataset (23 classes)
BALL_CLASS_MAPPING = {
    0: "bag1", 1: "bag2", 2: "bag3", 3: "bag4", 4: "bag5", 5: "bag6",
    6: "ball0", 7: "ball1", 8: "ball10", 9: "ball11", 10: "ball12", 11: "ball13",
    12: "ball14", 13: "ball15", 14: "ball2", 15: "ball3", 16: "ball4", 17: "ball5",
    18: "ball6", 19: "ball7", 20: "ball8", 21: "ball9", 22: "flag"
}'''
    
    # Find and replace the mapping section
    start_idx = content.find(old_mapping_start)
    if start_idx != -1:
        end_idx = content.find(old_mapping_end, start_idx) + 1
        content = content[:start_idx] + new_mapping + content[end_idx:]
    
    # Replace colors
    old_colors_start = "# Colors for different ball types"
    old_colors_end = "}"
    
    new_colors = '''# Colors for different ball types
BALL_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
    6: (255, 255, 255), 7: (255, 128, 0), 8: (128, 255, 0), 9: (0, 128, 255), 10: (255, 0, 128), 11: (128, 0, 255),
    12: (0, 255, 128), 13: (255, 128, 128), 14: (128, 128, 0), 15: (0, 128, 128), 16: (128, 0, 128), 17: (255, 165, 0),
    18: (0, 255, 0), 19: (128, 0, 0), 20: (0, 0, 0), 21: (255, 215, 0), 22: (255, 255, 255)
}'''
    
    start_idx = content.find(old_colors_start)
    if start_idx != -1:
        end_idx = content.find(old_colors_end, start_idx) + 1
        content = content[:start_idx] + new_colors + content[end_idx:]
    
    # Write updated content
    with open('src/detectors/multi_class_video_detector.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Video detector updated successfully!")

def print_dataset_info():
    """Print information about the new dataset"""
    
    print("📊 New Dataset Information:")
    print("=" * 40)
    print("Classes: 23")
    print("Names: ['bag1', 'bag2', 'bag3', 'bag4', 'bag5', 'bag6', 'ball0', 'ball1', 'ball10', 'ball11', 'ball12', 'ball13', 'ball14', 'ball15', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6', 'ball7', 'ball8', 'ball9', 'flag']")
    print("Total images: 4,521")
    print("Train: 3,905 images")
    print("Valid: 414 images") 
    print("Test: 202 images")
    print("Format: YOLOv11")
    print("Source: Roboflow Universe")

if __name__ == "__main__":
    print_dataset_info()
    print()
    update_video_detector()
    print("\n🎉 Class mapping update completed!") 
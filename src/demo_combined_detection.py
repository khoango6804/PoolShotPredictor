import cv2
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ball_detector import MultiObjectDetector
from src.models.pocket_detector import PocketDetector
from src.config.config import CLASSES, CLASS_COLORS

def create_synthetic_image():
    """
    Create a synthetic billiards table image for testing
    """
    # Create a green billiards table background
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)  # Forest green
    
    # Draw table border
    cv2.rectangle(img, (50, 50), (1230, 670), (139, 69, 19), 20)  # Brown border
    
    # Draw pockets at corners and middle of long sides
    pocket_positions = [
        (80, 80),      # Top-left
        (1200, 80),    # Top-right
        (80, 640),     # Bottom-left
        (1200, 640),   # Bottom-right
        (640, 80),     # Top-middle
        (640, 640)     # Bottom-middle
    ]
    
    for pos in pocket_positions:
        cv2.circle(img, pos, 25, (0, 0, 0), -1)  # Black pockets
    
    # Draw some balls
    ball_positions = [
        (400, 300, (255, 255, 255)),  # White ball
        (500, 350, (255, 0, 0)),      # Red ball
        (600, 400, (0, 0, 255)),      # Blue ball
        (450, 450, (0, 255, 255)),    # Yellow ball
        (550, 500, (255, 0, 255))     # Magenta ball
    ]
    
    for ball_data in ball_positions:
        x, y, color = ball_data
        cv2.circle(img, (x, y), 15, color, -1)
        cv2.circle(img, (x, y), 15, (0, 0, 0), 2)  # Black border
    
    return img

def load_test_image(image_path):
    """
    Load a test image from the dataset
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    return img

def run_detection_demo(model_path=None, image_path=None, use_synthetic=True):
    """
    Run detection demo with the combined model
    
    Args:
        model_path: Path to trained model (optional)
        image_path: Path to test image (optional)
        use_synthetic: Whether to use synthetic image for testing
    """
    print("Initializing Multi-Object Detector...")
    
    # Initialize detector
    detector = MultiObjectDetector(model_path=model_path)
    pocket_detector = PocketDetector()
    
    # Load or create test image
    if image_path and os.path.exists(image_path):
        print(f"Loading test image: {image_path}")
        img = load_test_image(image_path)
        if img is None:
            print("Failed to load image, using synthetic image instead")
            img = create_synthetic_image()
    elif use_synthetic:
        print("Creating synthetic test image...")
        img = create_synthetic_image()
    else:
        print("No image provided and synthetic disabled. Please provide an image path.")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Run detection
    print("Running detection...")
    detections = detector.detect(img)
    
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, class_id = det
        class_name = CLASSES.get(class_id, f"class_{class_id}")
        print(f"  {i+1}. {class_name}: conf={conf:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    
    # Run pocket detection
    print("Running pocket detection...")
    pocket_events = pocket_detector.update(detections, img)
    
    if pocket_events:
        print(f"Detected {len(pocket_events)} pocketing events:")
        for event in pocket_events:
            print(f"  Ball at {event['ball_position']} pocketed in {event['pocket_type']}")
    
    # Draw detections
    print("Drawing detections...")
    img_with_detections = detector.draw_detections(img.copy(), detections)
    
    # Draw pocket events
    if pocket_events:
        img_with_detections = pocket_detector.draw_pocket_events(img_with_detections, pocket_events)
    
    # Add statistics
    stats = pocket_detector.get_pocket_statistics()
    if stats:
        y_offset = 30
        for key, value in stats.items():
            if key != 'pocket_type_counts':
                cv2.putText(img_with_detections, f"{key}: {value}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
    
    # Save result
    output_path = "demo_result_combined.jpg"
    cv2.imwrite(output_path, img_with_detections)
    print(f"Result saved to: {output_path}")
    
    # Try to display (may not work in headless environments)
    try:
        cv2.imshow("Combined Detection Demo", img_with_detections)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Display not available: {e}")
        print(f"Result saved to: {output_path}")
    
    return detections, pocket_events

def test_with_dataset_images():
    """
    Test detection with actual dataset images
    """
    print("Testing with dataset images...")
    
    # Look for test images in datasets
    test_images = []
    
    # Check pocket detection dataset
    pocket_test_dir = Path("pocket detection/test/images")
    if pocket_test_dir.exists():
        test_images.extend(list(pocket_test_dir.glob("*.jpg"))[:3])
    
    # Check table detector dataset
    table_test_dir = Path("table detector/test/images")
    if table_test_dir.exists():
        test_images.extend(list(table_test_dir.glob("*.jpg"))[:3])
    
    # Check billiards dataset
    billiards_test_dir = Path("billiards-2/test/images")
    if billiards_test_dir.exists():
        test_images.extend(list(billiards_test_dir.glob("*.jpg"))[:3])
    
    if not test_images:
        print("No test images found in datasets")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test with first image
    test_image = test_images[0]
    print(f"Testing with: {test_image}")
    
    run_detection_demo(image_path=str(test_image), use_synthetic=False)

def main():
    parser = argparse.ArgumentParser(description='Combined Billiards Detection Demo')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--image', '-i', type=str, default=None,
                       help='Path to test image')
    parser.add_argument('--synthetic', '-s', action='store_true',
                       help='Use synthetic image for testing')
    parser.add_argument('--test-dataset', '-t', action='store_true',
                       help='Test with dataset images')
    
    args = parser.parse_args()
    
    if args.test_dataset:
        test_with_dataset_images()
    else:
        run_detection_demo(
            model_path=args.model,
            image_path=args.image,
            use_synthetic=args.synthetic
        )

if __name__ == "__main__":
    main() 
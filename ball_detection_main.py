#!/usr/bin/env python3
"""
Ball Detection Main Script
Script chính để detect bóng với multi-class approach
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detectors import detect_all_balls, process_video
from comparison import compare_results, process_video_single_class, process_video_multi_class

def main():
    parser = argparse.ArgumentParser(description="Multi-Class Ball Detection System")
    parser.add_argument("--mode", choices=["image", "video", "compare"], required=True,
                       help="Detection mode: image, video, or compare")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--model", default="yolov8m.pt",
                       help="Path to YOLO model")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--confidence", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="IOU threshold")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (video mode)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return
    
    print(f"🎯 Multi-Class Ball Detection System")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"=" * 50)
    
    if args.mode == "image":
        # Image detection
        print(f"\n🖼️ Processing image...")
        ball_types, original_result = detect_all_balls(args.model, args.input, args.confidence, args.iou)
        
        if ball_types:
            total_balls = sum(len(detections) for detections in ball_types.values())
            print(f"\n✅ Detection completed!")
            print(f"🎱 Total balls detected: {total_balls}")
            print(f"🎯 Ball types: {list(ball_types.keys())}")
            
            if args.output:
                print(f"💾 Results saved to: {args.output}")
        else:
            print(f"❌ No balls detected")
    
    elif args.mode == "video":
        # Video detection
        output_path = args.output or "ball_detection_output.mp4"
        print(f"\n🎬 Processing video...")
        
        stats = process_video(args.model, args.input, output_path, args.max_frames)
        
        if stats:
            print(f"\n✅ Video processing completed!")
            print(f"🎬 Output video: {output_path}")
            print(f"🎱 Total balls: {stats['total_balls']}")
            print(f"📈 Avg balls/frame: {stats['avg_balls_per_frame']:.1f}")
    
    elif args.mode == "compare":
        # Comparison mode
        print(f"\n🔍 Running comparison...")
        
        # Single class detection
        print(f"Step 1: Single-Class Detection")
        single_stats = process_video_single_class(args.model, args.input, args.max_frames)
        
        if single_stats is None:
            print("❌ Failed to process video with single-class detection")
            return
        
        # Multi class detection
        print(f"\nStep 2: Multi-Class Detection")
        multi_stats = process_video_multi_class(args.model, args.input, args.max_frames)
        
        if multi_stats is None:
            print("❌ Failed to process video with multi-class detection")
            return
        
        # Compare results
        comparison_results = compare_results(single_stats, multi_stats)
        
        # Save results
        output_file = args.output or "detection_comparison.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n💾 Comparison results saved to: {output_file}")
    
    print(f"\n✅ All operations completed successfully!")

if __name__ == "__main__":
    main() 
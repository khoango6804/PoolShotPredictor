"""
Comparison Tools Package
Các công cụ để so sánh hiệu quả giữa single-class và multi-class detection
"""

from .compare_video_detection import compare_results, process_video_single_class, process_video_multi_class

__all__ = [
    'compare_results',
    'process_video_single_class',
    'process_video_multi_class'
] 
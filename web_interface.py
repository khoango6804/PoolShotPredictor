#!/usr/bin/env python3
"""
Web interface for YOLOv11 billiards detection
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="YOLOv11 Billiards Detection",
    page_icon="🎱",
    layout="wide"
)

# Title
st.title("YOLOv11 Billiards Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.01, 1.0, 0.3, 0.01)
model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"

# Check if model exists
if not Path(model_path).exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    """Load YOLOv11 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'],
        help="Upload an image or video file to detect billiards balls"
    )

with col2:
    st.header("Model Info")
    st.info(f"**Model:** YOLOv11 Billiards Detection")
    st.info(f"**Classes:** 23 different ball types")
    st.info(f"**Confidence:** {confidence:.2f}")
    
    # Model statistics
    if Path(model_path).exists():
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        st.metric("Model Size", f"{model_size:.1f} MB")
    
    # Performance info
    st.info("**Performance:** GPU optimized for real-time detection")

# Detection function
def run_detection(image, conf_threshold):
    """Run YOLOv11 detection on image"""
    try:
        results = model(image, conf=float(conf_threshold), verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detections = len(boxes)
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                
                return {
                    'detections': detections,
                    'confidences': confidences,
                    'class_ids': class_ids,
                    'result_image': results[0].plot()
                }
        
        return None
    except Exception as e:
        st.error(f"Detection error: {e}")
        return None

# Process uploaded file
if uploaded_file is not None:
    st.markdown("---")
    st.header("Detection Results")
    
    # Get file info
    file_name = uploaded_file.name
    file_type = file_name.split('.')[-1].lower()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if file_type in ['jpg', 'jpeg', 'png']:
        # Process image
        status_text.text(" Processing image...")
        progress_bar.progress(25)
        
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        progress_bar.progress(50)
        status_text.text("Running detection...")
        
        # Run detection
        result = run_detection(image_array, confidence)
        
        progress_bar.progress(75)
        status_text.text("Saving results...")
        
        if result:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Original", use_column_width=True)
            
            with col2:
                st.subheader("Detection Result")
                st.image(result['result_image'], caption=f"Detections: {result['detections']}", use_column_width=True)
            
            # Statistics
            st.subheader("Detection Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Detections", result['detections'])
            
            with col2:
                if len(result['confidences']) > 0:
                    st.metric("Max Confidence", f"{result['confidences'].max():.3f}")
            
            with col3:
                if len(result['confidences']) > 0:
                    st.metric("Min Confidence", f"{result['confidences'].min():.3f}")
            
            with col4:
                if len(result['confidences']) > 0:
                    st.metric("Avg Confidence", f"{result['confidences'].mean():.3f}")
            
            # Class distribution
            if len(result['class_ids']) > 0:
                unique_classes, counts = np.unique(result['class_ids'].astype(int), return_counts=True)
                
                st.subheader("🏷️ Class Distribution")
                class_data = {f"Class {class_id}": count for class_id, count in zip(unique_classes, counts)}
                st.bar_chart(class_data)
                
                # Detailed class info
                st.subheader("Detailed Detections")
                for i, (class_id, conf) in enumerate(zip(result['class_ids'], result['confidences'])):
                    st.write(f"**Detection {i+1}:** Class {int(class_id)} - Confidence: {conf:.3f}")
        
        progress_bar.progress(100)
        status_text.text("Detection completed!")
        
    elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
        # Process video
        status_text.text("Processing video...")
        progress_bar.progress(25)
        
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        progress_bar.progress(50)
        status_text.text("Running video detection...")
        
        # Video processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Cannot open video file")
            os.unlink(video_path)
            st.stop()
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"**Video Info:** {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Process video (limit to first 100 frames for demo)
        max_frames = min(100, total_frames)
        processed_frames = 0
        total_detections = 0
        
        # Create output video writer
        output_path = f"output_{file_name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Progress tracking
        progress_text = st.empty()
        frame_progress = st.progress(0)
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            result = run_detection(frame, confidence)
            
            if result:
                # Write frame with detections
                out.write(result['result_image'])
                total_detections += result['detections']
            else:
                # Write original frame
                out.write(frame)
            
            processed_frames += 1
            
            # Update progress
            progress = processed_frames / max_frames
            frame_progress.progress(progress)
            progress_text.text(f"Processing frame {processed_frames}/{max_frames}")
        
        # Cleanup
        cap.release()
        out.release()
        os.unlink(video_path)
        
        progress_bar.progress(100)
        status_text.text("Video processing completed!")
        
        # Display results
        st.subheader("🎬 Video Processing Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processed Frames", processed_frames)
        
        with col2:
            st.metric("Total Detections", total_detections)
        
        with col3:
            if processed_frames > 0:
                st.metric("Avg Detections/Frame", f"{total_detections/processed_frames:.1f}")
        
        # Download processed video
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file.read(),
                file_name=f"detected_{file_name}",
                mime="video/mp4"
            )
        
        # Cleanup output file
        os.unlink(output_path)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>YOLOv11 Billiards Detection System</strong></p>
    <p>Powered by Ultralytics YOLOv11 | GPU Optimized</p>
</div>
""", unsafe_allow_html=True)

# Add some styling
st.markdown("""
<style>
    .main-header {
        background-color: #1f77b4;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True) 
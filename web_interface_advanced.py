#!/usr/bin/env python3
"""
Advanced Web Interface for YOLOv11 Billiards Detection
with Video and Real-time Controls
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
import threading
import queue
import pandas as pd

# Page config
st.set_page_config(
    page_title="YOLOv11 Advanced Detection",
    page_icon="🎱",
    layout="wide"
)

# Title
st.title("🎱 YOLOv11 Advanced Billiards Detection System")
st.markdown("---")

# Sidebar - Advanced Settings
st.sidebar.header(" Advanced Settings")

# Detection Settings
st.sidebar.subheader(" Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.01, 1.0, 0.3, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.5, 0.05)
max_detections = st.sidebar.number_input("Max Detections", 1, 100, 50, 1)

# Video Processing Settings
st.sidebar.subheader(" Video Processing")
video_fps = st.sidebar.number_input("Output FPS", 1, 60, 30, 1)
frame_skip = st.sidebar.number_input("Frame Skip", 1, 10, 1, 1, 
                                   help="Process every Nth frame")
max_frames = st.sidebar.number_input("Max Frames to Process", 10, 10000, 1000, 10)
resize_factor = st.sidebar.slider("Resize Factor", 0.1, 2.0, 1.0, 0.1,
                                help="Resize video frames (1.0 = original size)")

# Real-time Settings
st.sidebar.subheader(" Real-time Settings")
enable_realtime = st.sidebar.checkbox("Enable Real-time Mode", False)
realtime_fps = st.sidebar.number_input("Real-time FPS", 1, 30, 15, 1)
buffer_size = st.sidebar.number_input("Buffer Size", 1, 10, 3, 1)

# Model Selection
st.sidebar.subheader(" Model Selection")
model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"

# Check available models
available_models = {
    "Ball Detector": "runs/detect/yolo11_billiards_gpu/weights/best.pt",
    "Table Detector": "table_detector/yolo11_table_detector/weights/best.pt",
    "Pocket Detector": "pocket_detector/yolo11_pocket_detector/weights/best.pt"
}

# Filter available models
existing_models = {name: path for name, path in available_models.items() 
                  if Path(path).exists()}

if existing_models:
    selected_model_name = st.sidebar.selectbox(
        "Select Model", 
        list(existing_models.keys()),
        index=0
    )
    model_path = existing_models[selected_model_name]
else:
    st.sidebar.error("No models found!")
    st.stop()

# Load model
@st.cache_resource
def load_model(model_path):
    """Load YOLOv11 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None

model = load_model(model_path)

if model is None:
    st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header(" Upload & Process")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        [" Image Detection", " Video Processing", " Real-time Detection"],
        horizontal=True
    )

with col2:
    st.header(" Model Info")
    st.info(f"**Model:** {selected_model_name}")
    st.info(f"**Confidence:** {confidence:.2f}")
    st.info(f"**IoU:** {iou_threshold:.2f}")
    
    if Path(model_path).exists():
        model_size = Path(model_path).stat().st_size / (1024 * 1024)
        st.metric("Model Size", f"{model_size:.1f} MB")

# Detection function with advanced parameters
def run_detection_advanced(image, conf_threshold, iou_threshold, max_det):
    """Run YOLOv11 detection with advanced parameters"""
    try:
        results = model(
            image, 
            conf=float(conf_threshold), 
            iou=float(iou_threshold),
            max_det=int(max_det),
            verbose=False
        )
        
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
        st.error(f" Detection error: {e}")
        return None

# Image processing
if mode == " Image Detection":
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect billiards balls"
    )
    
    if uploaded_file is not None:
        st.markdown("---")
        st.header(" Image Detection Results")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(" Processing image...")
        progress_bar.progress(25)
        
        # Load and process image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if resize_factor != 1.0:
            h, w = image_array.shape[:2]
            new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            image_array = cv2.resize(image_array, (new_w, new_h))
        
        progress_bar.progress(50)
        status_text.text(" Running detection...")
        
        # Run detection
        result = run_detection_advanced(image_array, confidence, iou_threshold, max_detections)
        
        progress_bar.progress(75)
        status_text.text(" Saving results...")
        
        if result:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" Original Image")
                st.image(image, caption="Original", use_container_width=True)
            
            with col2:
                st.subheader(" Detection Result")
                st.image(result['result_image'], caption=f"Detections: {result['detections']}", use_container_width=True)
            
            # Advanced statistics
            st.subheader(" Advanced Detection Statistics")
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
            
            # Detailed analysis
            if len(result['class_ids']) > 0:
                unique_classes, counts = np.unique(result['class_ids'].astype(int), return_counts=True)
                
                st.subheader(" Class Distribution")
                class_data = {f"Class {class_id}": count for class_id, count in zip(unique_classes, counts)}
                st.bar_chart(class_data)
                
                # Detailed detections table
                st.subheader(" Detailed Detections")
                detection_data = []
                for i, (class_id, conf) in enumerate(zip(result['class_ids'], result['confidences'])):
                    detection_data.append({
                        "Detection": i+1,
                        "Class": int(class_id),
                        "Confidence": f"{conf:.3f}",
                        "Status": " High" if conf > 0.7 else "⚠️ Medium" if conf > 0.4 else "❌ Low"
                    })
                
                st.dataframe(detection_data, use_container_width=True)
        
        progress_bar.progress(100)
        status_text.text(" Detection completed!")

# Video processing
elif mode == " Video Processing":
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to process with detection"
    )
    
    if uploaded_file is not None:
        st.markdown("---")
        st.header(" Video Processing Results")
        
        # Video processing settings
        col1, col2, col3 = st.columns(3)
        with col1:
            show_progress = st.checkbox("Show Processing Progress", True)
        with col2:
            save_processed = st.checkbox("Save Processed Video", True)
        with col3:
            show_stats = st.checkbox("Show Frame Statistics", True)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(" Processing video...")
        progress_bar.progress(10)
        
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        progress_bar.progress(20)
        status_text.text(" Running video detection...")
        
        # Video processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(" Cannot open video file")
            os.unlink(video_path)
            st.stop()
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Apply resize factor
        if resize_factor != 1.0:
            width = int(width * resize_factor)
            height = int(height * resize_factor)
        
        st.info(f"**Video Info:** {width}x{height}, {original_fps:.1f} FPS, {total_frames} frames")
        st.info(f"**Processing:** Every {frame_skip} frame(s), Max {max_frames} frames, Output {video_fps} FPS")
        
        # Process video
        max_frames_to_process = min(max_frames, total_frames)
        processed_frames = 0
        total_detections = 0
        frame_stats = []
        start_time = time.time()  # Add start time
        
        # Create output video writer
        if save_processed:
            output_path = f"processed_{uploaded_file.name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        
        # Progress tracking
        frame_progress = st.progress(0)
        frame_text = st.empty()
        
        while processed_frames < max_frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            for _ in range(frame_skip - 1):
                cap.read()
            
            # Resize frame
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (width, height))
            
            # Run detection
            result = run_detection_advanced(frame, confidence, iou_threshold, max_detections)
            
            if result:
                # Write frame with detections
                if save_processed:
                    out.write(result['result_image'])
                total_detections += result['detections']
                
                # Store frame stats
                if show_stats:
                    frame_stats.append({
                        "Frame": processed_frames + 1,
                        "Detections": result['detections'],
                        "Avg Confidence": result['confidences'].mean() if len(result['confidences']) > 0 else 0
                    })
            else:
                # Write original frame
                if save_processed:
                    out.write(frame)
            
            processed_frames += 1
            
            # Update progress
            progress = processed_frames / max_frames_to_process
            frame_progress.progress(progress)
            frame_text.text(f"Processing frame {processed_frames}/{max_frames_to_process}")
            
            # Show real-time stats
            if show_progress and processed_frames % 10 == 0:
                st.write(f"Frame {processed_frames}: {result['detections'] if result else 0} detections")
        
        # Cleanup
        cap.release()
        if save_processed:
            out.release()
        os.unlink(video_path)
        
        progress_bar.progress(100)
        status_text.text("Video processing completed!")
        
        # Display results
        st.subheader("Video Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Processed Frames", processed_frames)
        
        with col2:
            st.metric("Total Detections", total_detections)
        
        with col3:
            if processed_frames > 0:
                st.metric("Avg Detections/Frame", f"{total_detections/processed_frames:.1f}")
        
        with col4:
            if processed_frames > 0:
                st.metric("Processing Speed", f"{processed_frames/(time.time() - start_time):.1f} fps")
        
        # Frame statistics chart
        if show_stats and frame_stats:
            st.subheader("Frame-by-Frame Statistics")
            df_stats = pd.DataFrame(frame_stats)
            st.line_chart(df_stats.set_index("Frame"))
        
        # Download processed video
        if save_processed and Path(output_path).exists():
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file.read(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="video/mp4"
                )
            
                    # Cleanup output file
        try:
            time.sleep(1)  # Wait a bit for file to be released
            if Path(output_path).exists():
                os.unlink(output_path)
        except PermissionError:
            st.warning(f"Could not delete temporary file: {output_path}")
        except Exception as e:
            st.warning(f"Error cleaning up file: {e}")

# Real-time detection
elif mode == " Real-time Detection":
    st.markdown("---")
    st.header(" Real-time Detection")
    
    if enable_realtime:
        st.warning("Real-time mode requires camera access")
        
        # Camera settings
        col1, col2 = st.columns(2)
        with col1:
            camera_source = st.selectbox("Camera Source", [0, 1, 2], help="Camera device index")
        with col2:
            show_fps = st.checkbox("Show FPS", True)
        
        # Real-time processing
        if st.button(" Start Real-time Detection"):
            st.subheader(" Live Detection")
            
            # Create placeholder for video
            video_placeholder = st.empty()
            
            # Start camera
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                st.error(" Cannot access camera")
                st.stop()
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FPS, realtime_fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Processing loop
            frame_count = 0
            start_time = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize frame
                    if resize_factor != 1.0:
                        h, w = frame.shape[:2]
                        new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                        frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Run detection
                    result = run_detection_advanced(frame, confidence, iou_threshold, max_detections)
                    
                    if result:
                        # Add FPS info
                        if show_fps:
                            fps = frame_count / (time.time() - start_time)
                            cv2.putText(result['result_image'], f"FPS: {fps:.1f}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Display result
                        video_placeholder.image(result['result_image'], channels="BGR", use_container_width=True)
                    else:
                        # Display original frame
                        if show_fps:
                            fps = frame_count / (time.time() - start_time)
                            cv2.putText(frame, f"FPS: {fps:.1f}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        video_placeholder.image(frame, channels="BGR", use_container_width=True)
                    
                    frame_count += 1
                    
                    # Add delay for FPS control
                    time.sleep(1/realtime_fps)
                    
            except KeyboardInterrupt:
                pass
            finally:
                cap.release()
    else:
        st.info(" Enable real-time mode in sidebar to start camera detection")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎱 <strong>YOLOv11 Advanced Billiards Detection System</strong></p>
    <p>Powered by Ultralytics YOLOv11 | GPU Optimized | Advanced Controls</p>
</div>
""", unsafe_allow_html=True)

# Add advanced styling
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
    .advanced-settings {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True) 
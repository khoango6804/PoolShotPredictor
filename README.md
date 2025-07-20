# 🎱 YOLOv11 Billiards Detection System

A comprehensive AI-powered billiards detection system using YOLOv11 for real-time ball, table, and pocket detection with web interface.

## 🌟 Features

### 🎯 Detection Capabilities
- **Ball Detection**: Detect and classify billiard balls in real-time
- **Table Detection**: Identify pool table boundaries and surfaces
- **Pocket Detection**: Locate table pockets for shot analysis
- **Multi-class Classification**: Support for different ball types and colors

### 🌐 Web Interface
- **Basic Interface**: Simple upload and detection (Port 8501)
- **Advanced Interface**: Full-featured with real-time camera, video processing, and detailed analytics (Port 8503)
- **Network Access**: Share interface across local network
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### ⚙️ Advanced Features
- **Real-time Processing**: Live camera feed with instant detection
- **Video Analysis**: Process uploaded videos with frame-by-frame analysis
- **Parameter Control**: Adjustable confidence, IoU, and detection thresholds
- **Statistics & Analytics**: Detailed performance metrics and detection statistics
- **Export Results**: Download processed images and videos

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/yolov11-billiards-detection.git
cd yolov11-billiards-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models**
```bash
# Models will be downloaded automatically on first run
python check_model_status.py
```

### Usage

#### Basic Interface (Simple)
```bash
python start_web_interface.py
# Access at: http://localhost:8501
```

#### Advanced Interface (Full Features)
```bash
python start_advanced_interface.py
# Access at: http://localhost:8503
```

#### Network Access (Share with others)
```bash
python start_network_simple.py
# Choose interface and get network URL
```

## 📁 Project Structure

```
poolShotPredictor/
├── src/                    # Source code
│   ├── models/            # Model definitions
│   ├── utils/             # Utility functions
│   └── config/            # Configuration files
├── web_interface.py       # Basic web interface
├── web_interface_advanced.py  # Advanced web interface
├── start_web_interface.py     # Basic interface launcher
├── start_advanced_interface.py # Advanced interface launcher
├── start_network_simple.py    # Network access launcher
├── check_model_status.py      # Model status checker
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Model Training

### Training Ball Detector
```bash
python train_ball_detector.py
```

### Training Table Detector
```bash
python train_table_detector.py
```

### Training Pocket Detector
```bash
python train_pocket_detector.py
```

## 🌐 Network Access

### Local Network Sharing
1. Run network launcher: `python start_network_simple.py`
2. Choose interface (Basic/Advanced)
3. Share the network URL with others on the same WiFi

### Example URLs
- **Local**: http://localhost:8501 (Basic)
- **Network**: http://192.168.0.101:8501 (Basic)
- **Local**: http://localhost:8503 (Advanced)
- **Network**: http://192.168.0.101:8503 (Advanced)

## 📊 Performance

- **Detection Speed**: ~30 FPS on RTX 4080
- **Accuracy**: 95%+ mAP50 on test dataset
- **Model Size**: ~5.2 MB (optimized)
- **Supported Formats**: JPG, PNG, MP4, AVI, MOV, MKV

## 🔧 Configuration

### Streamlit Configuration
Edit `.streamlit/config.toml` for custom settings:
```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Model Configuration
- Adjust confidence thresholds in web interface
- Modify detection parameters in `src/config/`
- Customize training settings in training scripts

## 📝 Usage Examples

### Image Detection
1. Open web interface
2. Upload image (JPG/PNG)
3. Adjust confidence threshold
4. View detection results
5. Download processed image

### Video Processing
1. Upload video file
2. Set processing parameters
3. Process video with detection
4. Download processed video

### Real-time Detection
1. Enable camera access
2. View live detection feed
3. Adjust parameters in real-time
4. Capture screenshots

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Streamlit](https://streamlit.io/) for web interface
- [OpenCV](https://opencv.org/) for computer vision
- [PyTorch](https://pytorch.org/) for deep learning

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/yolov11-billiards-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/yolov11-billiards-detection/discussions)
- **Email**: your.email@example.com

## 🔄 Updates

- **v1.0.0**: Initial release with basic detection
- **v1.1.0**: Added advanced interface and network access
- **v1.2.0**: Improved performance and added real-time features

---

⭐ **Star this repository if you find it useful!** 
import torch
import sys
import platform
import subprocess

def get_nvidia_smi():
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        return nvidia_smi.decode('utf-8')
    except:
        return "Không thể chạy nvidia-smi"

def check_cuda():
    print("=== Thông tin hệ thống ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print("\n=== Thông tin CUDA ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print("\n=== NVIDIA-SMI Output ===")
        print(get_nvidia_smi())
    else:
        print("CUDA không khả dụng. Kiểm tra:")
        print("1. Driver NVIDIA đã được cài đặt chưa")
        print("2. CUDA Toolkit đã được cài đặt chưa")
        print("3. PyTorch đã được cài đặt với CUDA support chưa")
        print("\nĐể cài đặt PyTorch với CUDA support, chạy lệnh:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_cuda() 
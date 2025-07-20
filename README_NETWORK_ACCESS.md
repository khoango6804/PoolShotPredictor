# 🌐 YOLOv11 BILLIARDS DETECTION - NETWORK ACCESS

## 🚀 Khởi động nhanh cho mọi người truy cập

### Phương pháp đơn giản nhất:
```bash
python start_network_simple.py
```

**Sau đó:**
1. Chọn interface (1 = Basic, 2 = Advanced)
2. Copy URL hiển thị
3. Chia sẻ cho mọi người trong mạng LAN

## 📱 Cách truy cập

### Từ máy chủ:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[IP]:8501

### Từ thiết bị khác:
- **Điện thoại/Tablet**: Mở trình duyệt → nhập Network URL
- **Máy tính khác**: Mở trình duyệt → nhập Network URL

## 🎯 Ví dụ thực tế

### Khi khởi động:
```
🎱 YOLOv11 Billiards Detection - Network Access
==================================================
🌐 Local IP: 192.168.1.100
🎯 Choose interface:
1. Basic Interface (Simple)
2. Advanced Interface (Full features)
Enter choice (1/2): 1
🚀 Starting Basic Interface...
📱 Local URL: http://localhost:8501
🌐 Network URL: http://192.168.1.100:8501
```

### Chia sẻ link:
- **Mọi người truy cập**: http://192.168.1.100:8501
- **Upload ảnh/video** từ điện thoại/máy tính
- **Xem kết quả detection** real-time

## 🔧 Troubleshooting nhanh

### Không truy cập được:
```bash
# Windows - Mở port
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501

# Tắt firewall tạm thời
# Kiểm tra cùng mạng WiFi/LAN
```

### Port bị chiếm:
- Script tự động tìm port khác
- Hoặc dừng process cũ

### Chậm:
- Sử dụng Basic Interface
- Giảm video quality
- Tăng frame skip

## 📊 Tính năng chia sẻ

### ✅ Mọi người có thể:
- Upload ảnh từ điện thoại
- Upload video từ máy tính
- Xem kết quả detection
- Download processed files
- Sử dụng real-time camera

### 🎮 Sử dụng:
- **Basic**: Dễ sử dụng, phù hợp mọi người
- **Advanced**: Đầy đủ tính năng, cho người dùng nâng cao

## 🔒 Bảo mật

### ⚠️ Lưu ý:
- **Chỉ chia sẻ trong mạng LAN tin cậy**
- **Không expose ra internet**
- **Tắt khi không sử dụng**

### 🛡️ An toàn:
- Chỉ mạng nội bộ
- Không lưu dữ liệu
- Tự động cleanup files

## 📞 Hỗ trợ

### Lệnh hữu ích:
```bash
# Khởi động network access
python start_network_simple.py

# Kiểm tra models
python check_model_status.py

# Test kết nối
curl http://localhost:8501
```

### Hướng dẫn chi tiết:
- `HUONG_DAN_NETWORK.md` - Hướng dẫn đầy đủ
- `HUONG_DAN_SU_DUNG.md` - Hướng dẫn sử dụng
- `README_QUICK_START.md` - Khởi động nhanh

---

## 🎉 Sẵn sàng chia sẻ!

**Chỉ cần 1 lệnh:**
```bash
python start_network_simple.py
```

**Mọi người trong mạng LAN có thể truy cập và sử dụng YOLOv11 Billiards Detection!** 🎱🌐✨ 
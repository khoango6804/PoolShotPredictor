# 🌐 HƯỚNG DẪN TRUY CẬP MẠNG - YOLOv11 BILLIARDS DETECTION

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Cách khởi động với network access](#cách-khởi-động-với-network-access)
3. [Truy cập từ thiết bị khác](#truy-cập-từ-thiết-bị-khác)
4. [Troubleshooting](#troubleshooting)
5. [Bảo mật](#bảo-mật)

## 🎯 Giới thiệu

Để mọi người trong mạng LAN có thể truy cập được giao diện web, bạn cần khởi động với cấu hình network access.

## 🚀 Cách khởi động với network access

### Phương pháp 1: Script Python đơn giản (Khuyến nghị)

```bash
python start_network_simple.py
```

**Cách sử dụng:**
1. Chạy script
2. Chọn interface (1 = Basic, 2 = Advanced)
3. Copy URL hiển thị
4. Chia sẻ URL cho mọi người

### Phương pháp 2: Script PowerShell (Windows)

```powershell
.\start_network.ps1
```

### Phương pháp 3: Script Bash (Linux/Mac)

```bash
chmod +x start_network.sh
./start_network.sh
```

### Phương pháp 4: Thủ công

```bash
# Basic Interface
streamlit run web_interface.py --server.port 8501 --server.address 0.0.0.0

# Advanced Interface  
streamlit run web_interface_advanced.py --server.port 8503 --server.address 0.0.0.0
```

## 📱 Truy cập từ thiết bị khác

### 1. Tìm IP của máy chủ

Khi khởi động, script sẽ hiển thị:
```
🌐 Local IP: 192.168.1.100
📱 Local URL: http://localhost:8501
🌐 Network URL: http://192.168.1.100:8501
```

### 2. Truy cập từ thiết bị khác

**Từ điện thoại/tablet:**
- Mở trình duyệt
- Nhập: `http://192.168.1.100:8501`
- Sử dụng giao diện như bình thường

**Từ máy tính khác:**
- Mở trình duyệt
- Nhập: `http://192.168.1.100:8501`
- Upload ảnh/video và xử lý

### 3. Chia sẻ link

**Link chia sẻ:**
- Basic: `http://192.168.1.100:8501`
- Advanced: `http://192.168.1.100:8503`

**Lưu ý:** Thay `192.168.1.100` bằng IP thực tế của máy chủ

## 🔧 Troubleshooting

### 1. Không truy cập được từ thiết bị khác

**Kiểm tra:**
```bash
# Kiểm tra firewall Windows
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501

# Kiểm tra kết nối
ping 192.168.1.100
```

**Giải pháp:**
- Tắt Windows Firewall tạm thời
- Kiểm tra antivirus có chặn không
- Đảm bảo cùng mạng WiFi/LAN

### 2. Port đã được sử dụng

**Giải pháp:**
- Script sẽ tự động tìm port khác
- Hoặc dừng process đang sử dụng port

### 3. Không tìm thấy IP

**Giải pháp:**
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

### 4. Kết nối chậm

**Tối ưu:**
- Giảm resolution video
- Tăng frame skip
- Sử dụng Basic Interface

## 🔒 Bảo mật

### Cảnh báo bảo mật:
- **Chỉ chia sẻ trong mạng LAN tin cậy**
- **Không expose ra internet**
- **Tắt khi không sử dụng**

### Cấu hình bảo mật:

**1. Giới hạn IP truy cập:**
```bash
# Chỉ cho phép IP cụ thể
streamlit run web_interface.py --server.address 192.168.1.100
```

**2. Sử dụng HTTPS (nâng cao):**
```bash
streamlit run web_interface.py --server.address 0.0.0.0 --server.sslCertFile cert.pem --server.sslKeyFile key.pem
```

**3. Authentication (nâng cao):**
- Thêm login system
- Sử dụng reverse proxy
- Cấu hình nginx

## 📊 Monitoring

### Kiểm tra kết nối:
```bash
# Xem active connections
netstat -an | findstr :8501

# Xem process
tasklist | findstr python
```

### Logs:
- Kiểm tra console output
- Xem Streamlit logs
- Monitor network traffic

## 🎯 Tips sử dụng

### Cho hiệu suất tốt:
1. **Sử dụng Basic Interface** cho nhiều người dùng
2. **Giảm video quality** nếu cần
3. **Tăng frame skip** cho video dài
4. **Sử dụng cable LAN** thay vì WiFi

### Cho trải nghiệm tốt:
1. **Hướng dẫn người dùng** cách upload file
2. **Giải thích các tham số** quan trọng
3. **Cung cấp ví dụ** ảnh/video mẫu
4. **Hỗ trợ real-time** khi cần

## 📞 Hỗ trợ

### Lệnh hữu ích:
```bash
# Kiểm tra network
python start_network_simple.py

# Test kết nối
curl http://localhost:8501

# Xem IP
ipconfig /all
```

### Liên hệ:
- Kiểm tra logs trong terminal
- Xem error messages
- Test với file nhỏ trước

---

## 🎉 Sẵn sàng chia sẻ!

Với cấu hình network access, mọi người trong mạng LAN có thể:
- ✅ Truy cập từ điện thoại/tablet
- ✅ Upload ảnh/video từ xa
- ✅ Xem kết quả detection
- ✅ Download processed files
- ✅ Sử dụng real-time detection

**Chia sẻ link và tận hưởng!** 🎱🌐 
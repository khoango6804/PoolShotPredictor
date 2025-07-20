# YOLOv11 Billiards Detection - Network Access Script (Windows)
Write-Host "🎱 YOLOv11 Billiards Detection - Network Access" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# Get local IP address
$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" -or $_.IPAddress -like "172.*"} | Select-Object -First 1).IPAddress
if (-not $LocalIP) {
    $LocalIP = "127.0.0.1"
}

Write-Host "🌐 Local IP: $LocalIP" -ForegroundColor Cyan

# Check if port 8501 is available
$Port8501 = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($Port8501) {
    Write-Host "⚠️ Port 8501 is in use, using port 8502" -ForegroundColor Yellow
    $Port = 8502
} else {
    $Port = 8501
}

Write-Host "🚀 Starting web interface on port $Port" -ForegroundColor Green
Write-Host "📱 Local URL: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "🌐 Network URL: http://$LocalIP:$Port" -ForegroundColor Cyan
Write-Host "⏹️ Press Ctrl+C to stop" -ForegroundColor Red
Write-Host "==============================================" -ForegroundColor Green

# Start Streamlit with network access
try {
    python -m streamlit run web_interface.py --server.port $Port --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
} catch {
    Write-Host "❌ Error starting web interface: $_" -ForegroundColor Red
} 
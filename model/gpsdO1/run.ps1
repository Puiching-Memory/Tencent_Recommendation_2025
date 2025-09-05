# GPSD Training Script for Windows
# Simple launcher that delegates to the unified Python launcher

Write-Host "=== GPSD Training for TencentGR_1k Dataset ===" -ForegroundColor Cyan
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host

# Install Python dependencies
pip install -r requirements.txt

Write-Host
Write-Host "Dependencies installed successfully!" -ForegroundColor Green
Write-Host "Delegating to unified launcher..."
Write-Host

# Pass all arguments to the Python launcher
& python train.py $args

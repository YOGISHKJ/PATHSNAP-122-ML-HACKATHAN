Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " PathSnap Wi-Fi Network Hosting Launcher" -ForegroundColor Cyan
Write-Host " Your Wi-Fi IP : 10.8.121.23" -ForegroundColor Yellow
Write-Host " FastAPI API   : http://10.8.121.23:8000" -ForegroundColor Cyan
Write-Host " Streamlit UI  : http://10.8.121.23:8501" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "$PWD;$PWD\src;$PWD\route_reconstruction"

# Start Backend on 0.0.0.0 (all interfaces)
Start-Process cmd.exe -ArgumentList '/k', 'python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000'
Start-Sleep -Seconds 3

# Start Frontend on 0.0.0.0 (all interfaces)
Start-Process cmd.exe -ArgumentList '/k', 'streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0'

Write-Host "`n[SUCCESS] PathSnap is now live on your Wi-Fi network!" -ForegroundColor Green
Write-Host "Share these links with your teammates on the same Wi-Fi:" -ForegroundColor White
Write-Host "  -> Interactive Web UI : http://10.8.121.23:8501" -ForegroundColor Yellow
Write-Host "  -> REST API Docs      : http://10.8.121.23:8000/docs" -ForegroundColor Yellow
Write-Host "`nNote: If they cannot connect, make sure Windows Firewall allows inbound connections on ports 8000 and 8501." -ForegroundColor Gray

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " PathSnap Single-Laptop Development Launcher" -ForegroundColor Cyan
Write-Host " Starting Backend on :8000 and Frontend on :8501" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "$PWD;$PWD\src;$PWD\route_reconstruction"

Start-Process cmd.exe -ArgumentList '/k', 'python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000'
Start-Sleep -Seconds 3
Start-Process cmd.exe -ArgumentList '/k', 'streamlit run frontend/app.py --server.port 8501'

Write-Host "Services launched successfully in separate windows." -ForegroundColor Green
Write-Host "Backend API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Streamlit UI:     http://localhost:8501" -ForegroundColor Yellow

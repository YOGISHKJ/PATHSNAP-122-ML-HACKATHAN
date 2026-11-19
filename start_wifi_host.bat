@echo off
echo ========================================================
echo  PathSnap Wi-Fi Network Hosting Launcher
echo  Your Wi-Fi IP: 10.8.121.23
echo  FastAPI Backend : http://10.8.121.23:8000
echo  Streamlit UI    : http://10.8.121.23:8501
echo ========================================================
set PYTHONPATH=%cd%;%cd%\src;%cd%\route_reconstruction

start "PathSnap Backend (Wi-Fi 0.0.0.0:8000)" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak
start "PathSnap Frontend (Wi-Fi 0.0.0.0:8501)" cmd /k "streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"

echo.
echo [SUCCESS] Services are now live on your Wi-Fi network!
echo Teammates on the same Wi-Fi can open:
echo   --^> Web Dashboard: http://10.8.121.23:8501
echo   --^> API Docs     : http://10.8.121.23:8000/docs
echo.

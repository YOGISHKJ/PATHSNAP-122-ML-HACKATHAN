@echo off
echo ========================================================
echo  PathSnap Single-Laptop Development Launcher
echo  Starting Backend on :8000 and Frontend on :8501
echo ========================================================
set PYTHONPATH=%cd%;%cd%\src;%cd%\route_reconstruction
start "PathSnap FastAPI Backend" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak
start "PathSnap Streamlit Frontend" cmd /k "streamlit run frontend/app.py --server.port 8501"
echo Services launched.

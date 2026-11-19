@echo off
echo ========================================================
echo  Starting PathSnap FastAPI Backend Service (Laptop 1)
echo  Listening on: http://127.0.0.1:8000
echo ========================================================
set PYTHONPATH=%cd%;%cd%\src;%cd%\route_reconstruction
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause

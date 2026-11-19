@echo off
echo ========================================================
echo  Starting PathSnap Streamlit Frontend (Laptop 2)
echo  Opening on: http://127.0.0.1:8501
echo  Connecting to Backend at: %BACKEND_URL% (default http://127.0.0.1:8000)
echo ========================================================
set PYTHONPATH=%cd%;%cd%\src;%cd%\route_reconstruction
if "%BACKEND_URL%"=="" set BACKEND_URL=http://127.0.0.1:8000
streamlit run frontend/app.py --server.port 8501 --server.headless false
pause

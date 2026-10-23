@echo off
cd /d "%~dp0"
python -c "import streamlit, cryptography, matplotlib, pandas, PIL, cv2, imageio_ffmpeg; v=tuple(int(x) for x in streamlit.__version__.split('.')[:2]); assert v>=(1,40)" 2>nul
if errorlevel 1 (
  echo Installing or updating the MAEO Studio packages. This needs internet and takes a few minutes...
  python -m pip install -r requirements.txt
)
echo.
echo Starting MAEO Studio... your browser will open at http://localhost:8501
echo Keep this window open while you use the app.
python -m streamlit run app.py --server.port 8501
pause

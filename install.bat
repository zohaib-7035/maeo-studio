@echo off
cd /d "%~dp0"
echo Installing MAEO Studio (needs internet)...
python -m pip install -r requirements.txt
echo.
echo Done. Now double-click run.bat
pause

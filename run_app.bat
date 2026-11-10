@echo off
REM Thermal Digital Twin demo launcher for Windows
REM Double-click to start. The app opens in your web browser.
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto :have_venv

echo [setup] Creating virtual environment .venv ...
where py >nul 2>nul
if errorlevel 1 goto :use_python
REM prefer Python versions that all dependencies ship wheels for
py -3.12 -m venv .venv >nul 2>nul
if exist "%PY%" goto :venv_made
py -3.13 -m venv .venv >nul 2>nul
if exist "%PY%" goto :venv_made
py -3.11 -m venv .venv >nul 2>nul
if exist "%PY%" goto :venv_made
py -3 -m venv .venv
goto :venv_made
:use_python
python -m venv .venv
:venv_made
if not exist "%PY%" goto :fail_venv
echo [setup] Installing packages - this takes a few minutes the first time ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail_pip

:have_venv
if exist "data\twin.json" goto :run_app
echo [setup] Building the digital twin: scan, SAM3, mesh, twin - about 30 seconds ...
"%PY%" scripts\build_twin.py
if errorlevel 1 goto :fail_build

:run_app
echo.
echo  Thermal Digital Twin demo is starting.
echo  The browser opens automatically. If not, go to http://127.0.0.1:8501
echo  To quit, close this window or press Ctrl+C.
echo.
"%PY%" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
if errorlevel 1 goto :fail_app
goto :eof

:fail_venv
echo.
echo [ERROR] Could not create the virtual environment. Install Python 3.11-3.13 from python.org and try again.
pause
goto :eof
:fail_pip
echo.
echo [ERROR] Package installation failed. Check the messages above - internet connection?
pause
goto :eof
:fail_build
echo.
echo [ERROR] Building the digital twin failed. See the messages above.
pause
goto :eof
:fail_app
echo.
echo [ERROR] The app stopped with an error. See the messages above.
pause
goto :eof

@echo off
REM Setup script for Advanced CV System (Windows)

echo Setting up Advanced Computer Vision System...

REM Check Python version
python --version
if errorlevel 1 (
    echo Error: Python not found!
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Create output directory
if not exist "output" mkdir output
echo Created output directory

REM Check for CUDA
echo.
echo Checking for CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo.
echo Setup complete!
echo.
echo To run the system:
echo   venv\Scripts\activate
echo   python src\main.py
echo.
echo For help:
echo   python src\main.py --help

pause

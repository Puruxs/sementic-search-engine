@echo off
title Private AI Search & Book/Document Engine
echo ========================================================
echo   Starting Private AI Search & Book/Document Engine
echo ========================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check core packages
python -c "import streamlit, pymupdf, docx, openpyxl, pandas, ddgs" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages...
    python -m pip install streamlit ddgs pymupdf pypdf python-docx openpyxl pandas pillow pytesseract requests beautifulsoup4 ollama groq python-dotenv
)

echo.
echo ========================================================
echo   Launching application on: http://localhost:8501
echo ========================================================
echo.

python -m streamlit run app.py --server.port 8501

pause

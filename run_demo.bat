@echo off
echo ========================================
echo Dataset Quality Auditor - Starting...
echo ========================================
echo.
echo Opening in your browser...
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
call venv\Scripts\activate
streamlit run app.py --server.port 8501 --server.headless false

pause
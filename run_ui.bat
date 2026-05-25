@echo off
echo ================================================
echo  Face Anonymizer UI 실행
echo ================================================
echo.

REM Streamlit이 설치되어 있는지 확인
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [설치] Streamlit UI 의존성 설치 중...
    pip install -r requirements_ui.txt
)

echo [실행] Streamlit UI 시작...
echo  브라우저에서 http://localhost:8501 접속하세요
echo.
streamlit run app.py --server.port 8501 --server.headless false

pause

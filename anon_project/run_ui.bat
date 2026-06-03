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

REM Ultralytics가 설치되어 있는지 확인
python -c "import ultralytics" 2>nul
if %errorlevel% neq 0 (
    echo [설치] Ultralytics 패키지 설치 중...
    pip install ultralytics
)

REM Diffusers 등 백엔드 의존성이 설치되어 있는지 확인
python -c "import diffusers" 2>nul
if %errorlevel% neq 0 (
    echo [설치] 딥러닝 백엔드(Face Anonymizer) 패키지 설치 중...
    pip install -r face-anonymizer\requirements.txt
)

echo [실행] Streamlit UI 시작...
echo  브라우저에서 http://localhost:8501 접속하세요
echo.
streamlit run app.py --server.port 8501 --server.headless false

pause

@echo off
echo Setting up AIS LLM Project...
echo.

if not exist venv\Scripts\activate.bat (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Make sure Python 3.12+ is installed.
        exit /b 1
    )
)

echo 1. Activating Python virtual environment...
call venv\Scripts\activate.bat

echo 2. Installing Python dependencies...
pip install -r requirements.txt

echo 3. Preprocessing AIS data (skipped if backend\data\AIS_2022_03_31.csv is missing)...
if exist backend\data\AIS_2022_03_31.csv (
    python -m backend.scripts.load_data
) else (
    echo    [skip] Raw AIS CSV not found; using committed cleaned_data.csv if present.
)

echo 4. Installing Node.js dependencies for React dashboard...
cd dashboard-app
call npm install
cd ..

echo.
echo Setup complete!
echo.
echo To start the application:
echo 1. Backend:  python -m uvicorn backend.main:app --reload --port 8000
echo 2. Frontend: cd dashboard-app ^&^& npm start
echo.
echo Then visit:
echo - Backend API:    http://127.0.0.1:8000/
echo - API docs:       http://127.0.0.1:8000/docs
echo - React Dashboard: http://localhost:3000/
echo.
pause
@echo off
echo Setting up AIS LLM Project...
echo.

echo 1. Activating Python virtual environment...
call venv\Scripts\activate.bat

echo 2. Installing Python dependencies...
pip install -r requirements.txt

echo 3. Processing AIS data...
python backend\scripts\data_processing_simple.py

echo 4. Installing Node.js dependencies for React dashboard...
cd dashboard-app
call npm install
cd ..

echo.
echo Setup complete!
echo.
echo To start the application:
echo 1. Backend: uvicorn backend.main:app --reload
echo 2. Frontend: cd dashboard-app && npm start
echo.
echo Then visit:
echo - Backend API: http://127.0.0.1:8000/
echo - React Dashboard: http://localhost:3000/
echo.
pause
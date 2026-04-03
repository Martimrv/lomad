Write-Host "Setting up AIS LLM Project..." -ForegroundColor Green
Write-Host ""

Write-Host "1. Activating Python virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host "2. Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "3. Processing AIS data..." -ForegroundColor Yellow
python backend\scripts\data_processing_simple.py

Write-Host "4. Installing Node.js dependencies for React dashboard..." -ForegroundColor Yellow
Set-Location dashboard-app
npm install
Set-Location ..

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Backend: uvicorn backend.main:app --reload" -ForegroundColor White
Write-Host "2. Frontend: cd dashboard-app; npm start" -ForegroundColor White
Write-Host ""
Write-Host "Then visit:" -ForegroundColor Cyan
Write-Host "- Backend API: http://127.0.0.1:8000/" -ForegroundColor White
Write-Host "- React Dashboard: http://localhost:3000/" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"
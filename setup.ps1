Write-Host "Setting up AIS LLM Project..." -ForegroundColor Green
Write-Host ""

if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create venv. Make sure Python 3.12+ is installed." -ForegroundColor Red
        exit 1
    }
}

Write-Host "1. Activating Python virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host "2. Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "3. Preprocessing AIS data (skipped if backend\data\AIS_2022_03_31.csv is missing)..." -ForegroundColor Yellow
if (Test-Path "backend\data\AIS_2022_03_31.csv") {
    python -m backend.scripts.load_data
} else {
    Write-Host "   [skip] Raw AIS CSV not found; using committed cleaned_data.csv if present." -ForegroundColor DarkYellow
}

Write-Host "4. Installing Node.js dependencies for React dashboard..." -ForegroundColor Yellow
Set-Location dashboard-app
npm install
Set-Location ..

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Backend:  python -m uvicorn backend.main:app --reload --port 8000" -ForegroundColor White
Write-Host "2. Frontend: cd dashboard-app; npm start" -ForegroundColor White
Write-Host ""
Write-Host "Then visit:" -ForegroundColor Cyan
Write-Host "- Backend API:     http://127.0.0.1:8000/" -ForegroundColor White
Write-Host "- API docs:        http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "- React Dashboard: http://localhost:3000/" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"
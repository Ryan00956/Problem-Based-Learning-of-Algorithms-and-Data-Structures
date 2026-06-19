param(
    [string]$Python = "",
    [string]$Requirements = "requirements.txt"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot $Requirements

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $VenvPython)) {
        if ($Python) {
            & $Python -m venv $VenvDir
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m venv $VenvDir
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv $VenvDir
        } else {
            throw "Python 3.10+ was not found. Install Python, then rerun this script."
        }
    }

    & $VenvPython -m pip install --upgrade pip
    if (-not (Test-Path $RequirementsPath)) {
        throw "Requirements file was not found: $RequirementsPath"
    }

    & $VenvPython -m pip install -r $RequirementsPath

    Write-Host "Virtual environment is ready: $VenvDir"
    Write-Host "Installed dependencies from: $Requirements"
    Write-Host "Run commands with .\run.ps1 or start the dashboard with .\start_frontend.ps1"
} finally {
    Pop-Location
}

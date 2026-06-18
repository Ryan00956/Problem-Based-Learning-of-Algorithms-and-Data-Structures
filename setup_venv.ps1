param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

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
    & $VenvPython -m pip install -r $Requirements

    Write-Output "Virtual environment is ready: $VenvDir"
    Write-Output "Run commands with .\run.ps1 or start the dashboard with .\start_frontend.ps1"
} finally {
    Pop-Location
}

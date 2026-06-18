param(
    [int]$Port = 8013
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $Python = "py"
} else {
    throw "Python was not found. Run .\setup_venv.ps1 after installing Python 3.10+."
}

Push-Location $ProjectRoot
try {
    & .\export_frontend_data.ps1
    $listening = $false
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2
        $listening = $response.StatusCode -eq 200
    } catch {
        $listening = $false
    }

    if (-not $listening) {
        Start-Process -FilePath $Python -ArgumentList @("-m", "http.server", "$Port", "--directory", "web") -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }

    Write-Output "Frontend URL: http://127.0.0.1:$Port/"
} finally {
    Pop-Location
}

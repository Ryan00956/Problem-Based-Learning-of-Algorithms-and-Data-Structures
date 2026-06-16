param(
    [int]$Port = 8013
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\MECHREVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
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

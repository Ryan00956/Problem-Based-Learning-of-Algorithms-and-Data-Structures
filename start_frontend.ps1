param(
    [int]$Port = 8013,
    [string]$Dataset = "movielens"
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
    try {
        & $Python -c "import fastapi, uvicorn" | Out-Null
    } catch {
        throw "FastAPI dependencies are missing. Run .\setup_venv.ps1 to install requirements.txt."
    }

    $listening = $false
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 2
        $listening = $response.StatusCode -eq 200
    } catch {
        $listening = $false
    }

    if (-not $listening) {
        $portBusy = $false
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $portBusy = $true
        } catch {
            $portBusy = $false
        }
        if ($portBusy) {
            throw "Port $Port is already serving a non-API frontend. Stop that process or choose another port."
        }
        Start-Process -FilePath $Python -ArgumentList @("-m", "src.api", "--port", "$Port", "--dataset", "$Dataset") -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        for ($attempt = 0; $attempt -lt 20; $attempt += 1) {
            Start-Sleep -Milliseconds 500
            try {
                $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) {
                    $listening = $true
                    break
                }
            } catch {
                $listening = $false
            }
        }
        if (-not $listening) {
            throw "FastAPI server did not become ready on port $Port."
        }
    }

    Write-Output "Frontend URL: http://127.0.0.1:$Port/"
    Write-Output "API Health: http://127.0.0.1:$Port/api/health"
    Write-Output "Dataset: $Dataset"
} finally {
    Pop-Location
}

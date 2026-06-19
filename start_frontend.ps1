param(
    [int]$Port = 8013,
    [string]$Dataset = "movielens",
    [switch]$NoBrowser,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SetupScript = Join-Path $ProjectRoot "setup_venv.ps1"

function Ensure-DemoEnvironment {
    if (-not (Test-Path $VenvPython)) {
        if ($SkipInstall) {
            throw "Local virtual environment was not found. Run .\setup_venv.ps1, or rerun without -SkipInstall."
        }
        Write-Host "Local virtual environment was not found. Creating it now..."
        & $SetupScript
    }
}

Ensure-DemoEnvironment

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $Python = "py"
} else {
    throw "Python was not found. Run .\setup_venv.ps1 after installing Python 3.10+."
}

function Stop-ProjectApiOnPort {
    param(
        [int]$Port
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.State -in @("Listen", "Bound") } |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($processId in $connections) {
        if ($processId -eq $PID) {
            continue
        }

        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
        if (-not $process -or -not $process.CommandLine) {
            continue
        }

        $isProjectApi = $process.CommandLine -like "*src.api*" -and $process.CommandLine -like "*--port*" -and $process.CommandLine -like "*$Port*"
        if (-not $isProjectApi) {
            continue
        }

        Write-Host "Stopping existing backend process $processId on port $Port..."
        Stop-Process -Id $processId -Force -ErrorAction Stop

        for ($attempt = 0; $attempt -lt 20; $attempt += 1) {
            Start-Sleep -Milliseconds 250
            $stillRunning = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if (-not $stillRunning) {
                break
            }
        }
    }
}

Push-Location $ProjectRoot
try {
    try {
        & $Python -c "import pandas, numpy, fastapi, uvicorn, duckdb" | Out-Null
    } catch {
        if ($SkipInstall) {
            throw "Python dependencies are missing. Run .\setup_venv.ps1, or rerun without -SkipInstall."
        }
        Write-Host "Python dependencies are missing. Installing the demo requirements now..."
        & $SetupScript
        $Python = $VenvPython
        & $Python -c "import pandas, numpy, fastapi, uvicorn, duckdb" | Out-Null
    }

    Stop-ProjectApiOnPort -Port $Port

    $listening = $false
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 2
        $listening = $response.StatusCode -eq 200
    } catch {
        $listening = $false
    }

    if ($listening) {
        throw "Port $Port is already serving an API that was not started by this script. Stop that process or choose another port."
    }

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

    $FrontendUrl = "http://127.0.0.1:$Port/"
    $HealthUrl = "http://127.0.0.1:$Port/api/health"
    if (-not $NoBrowser) {
        $BrowserOpenScript = @"
`$url = "$FrontendUrl"
`$health = "$HealthUrl"
for (`$attempt = 0; `$attempt -lt 40; `$attempt += 1) {
    try {
        `$response = Invoke-WebRequest -Uri `$health -UseBasicParsing -TimeoutSec 2
        if (`$response.StatusCode -eq 200) {
            Start-Process `$url
            break
        }
    } catch {
    }
    Start-Sleep -Milliseconds 500
}
"@
        Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $BrowserOpenScript) -WindowStyle Hidden
    }

    Write-Host "Frontend URL: $FrontendUrl"
    Write-Host "API Health: $HealthUrl"
    Write-Host "Dataset: $Dataset"
    Write-Host "Backend is running in this terminal. Press Ctrl+C or close this window to stop it."
    $BackendProcess = $null
    try {
        $BackendProcess = Start-Process -FilePath $Python -ArgumentList @("-m", "src.api", "--port", "$Port", "--dataset", "$Dataset") -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru
        Wait-Process -Id $BackendProcess.Id
        $BackendProcess.Refresh()
        if ($BackendProcess.ExitCode -ne 0) {
            exit $BackendProcess.ExitCode
        }
    } finally {
        if ($BackendProcess -and -not $BackendProcess.HasExited) {
            Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
        }
    }
} finally {
    Pop-Location
}

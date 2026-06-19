$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SetupScript = Join-Path $ProjectRoot "setup_venv.ps1"

if (-not (Test-Path $VenvPython) -and (Test-Path $SetupScript)) {
    Write-Host "Local virtual environment was not found. Creating it now..."
    & $SetupScript
}

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
        & $Python -c "import pandas, numpy, fastapi, uvicorn, duckdb" | Out-Null
    } catch {
        if (-not (Test-Path $SetupScript)) {
            throw "Python dependencies are missing. Install requirements.txt, then rerun this command."
        }
        Write-Host "Python dependencies are missing. Installing the demo requirements now..."
        & $SetupScript
        $Python = $VenvPython
    }

    $Dataset = "movielens"
    for ($index = 0; $index -lt $args.Count; $index += 1) {
        if ($args[$index] -eq "--dataset" -and ($index + 1) -lt $args.Count) {
            $Dataset = $args[$index + 1]
            break
        }
    }

    & $Python -m src.bootstrap_data --dataset $Dataset
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $Python -m src.main @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

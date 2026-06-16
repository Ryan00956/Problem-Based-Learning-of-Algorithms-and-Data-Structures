$ErrorActionPreference = "Stop"

$Python = "C:\Users\MECHREVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m src.export_frontend_data

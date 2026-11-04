$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
& "$projectRoot\.venv\Scripts\python.exe" scripts/dev.py
exit $LASTEXITCODE

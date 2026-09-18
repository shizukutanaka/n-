# nmesh installer for Windows — creates a dedicated venv and installs
# nmesh with the gateway + download extras.
$ErrorActionPreference = "Stop"

$Venv = if ($env:NMESH_VENV) { $env:NMESH_VENV } else { "$env:USERPROFILE\.nmesh\venv" }
$Repo = if ($env:NMESH_REPO) { $env:NMESH_REPO } else { "https://github.com/shizukutanaka/n-.git" }

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "python is required (>= 3.10). Install it from https://www.python.org/"
    exit 1
}
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "python >= 3.10 is required"
    exit 1
}

python -m venv $Venv
& "$Venv\Scripts\pip.exe" install --upgrade pip
& "$Venv\Scripts\pip.exe" install "nmesh[gateway,download] @ git+$Repo"

Write-Host ""
Write-Host "nmesh installed into $Venv"
Write-Host "Add it to your PATH:"
Write-Host "  `$env:PATH = `"$Venv\Scripts;`$env:PATH`""
Write-Host ""
Write-Host "Then:"
Write-Host "  nmesh doctor      # check detected hardware and backends"
Write-Host "  nmesh up --detach # pick models for this machine and start the gateway"

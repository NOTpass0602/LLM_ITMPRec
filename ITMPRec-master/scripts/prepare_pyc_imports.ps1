$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Pycache = Join-Path $RepoRoot "src\__pycache__"

$Files = @(
    @{ Source = "modules.cpython-38.pyc"; Target = "modules.pyc" },
    @{ Source = "trainers.cpython-38.pyc"; Target = "trainers.pyc" }
)

foreach ($File in $Files) {
    $Source = Join-Path $Pycache $File.Source
    $Target = Join-Path (Join-Path $RepoRoot "src") $File.Target
    if (!(Test-Path $Source)) {
        throw "Missing $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Target -Force
    Write-Host "Prepared $Target"
}

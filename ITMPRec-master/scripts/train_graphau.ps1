param(
    [ValidateSet("ml100k", "lastfm", "steam", "douban_movie")]
    [string]$Dataset = "ml100k",
    [int]$Gpu = 0,
    [int]$EmbedSize = 64,
    [int]$Epoch = 999999,
    [int]$BatchSize = 2048,
    [int]$Layers = 2,
    [double]$Lr = 0.1,
    [double]$WeightDecay = 1e-6,
    [double]$GammaAU = 0.4,
    [double]$AlphaN = 0.1,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$GraphAURoot = (Resolve-Path (Join-Path $RepoRoot "..\GraphAU-main\GraphAU-main")).Path
$DataRoot = (Resolve-Path (Join-Path $RepoRoot "data_graphau\datasets")).Path
$ExportDir = Join-Path $RepoRoot "data\$Dataset"

$env:DGLBACKEND = "pytorch"
$env:GRAPHAU_RUN_DIR = Join-Path $RepoRoot "graphau_runs"
if ([string]::IsNullOrWhiteSpace($Python)) {
    if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    }
    else {
        $Python = "python"
    }
}
Write-Host "Using Python: $Python"

New-Item -ItemType Directory -Force -Path (Join-Path $env:GRAPHAU_RUN_DIR "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $env:GRAPHAU_RUN_DIR "best_models") | Out-Null

Push-Location $GraphAURoot
try {
    & $Python main.py `
        --dataset $Dataset `
        --data_root $DataRoot `
        --model graphau `
        --embed_size $EmbedSize `
        --lr $Lr `
        --weight_decay $WeightDecay `
        --layers $Layers `
        --gamma_au $GammaAU `
        --alpha_n $AlphaN `
        --batch_size $BatchSize `
        --epoch $Epoch `
        --gpu $Gpu `
        --export_dir $ExportDir
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

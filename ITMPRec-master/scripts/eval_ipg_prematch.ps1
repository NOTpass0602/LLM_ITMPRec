param(
    [ValidateSet("ml100k", "lastfm", "steam", "douban_movie")]
    [string]$Dataset = "ml100k",
    [int]$Gpu = 0,
    [int]$ModelIdx = 1,
    [int]$Clusters = 32,
    [int]$BatchSize = 256,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$Counts = @{
    ml100k       = @{ Users = 943; Items = 1682 }
    lastfm       = @{ Users = 1892; Items = 12523 }
    steam        = @{ Users = 68403; Items = 10050 }
    douban_movie = @{ Users = 2712; Items = 34893 }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SrcRoot = Join-Path $RepoRoot "src"
$DataDir = "..\data\$Dataset\"
$UserEmbPath = "..\data\$Dataset\user_emb-v2-64.pt"
$ItemEmbPath = "..\data\$Dataset\item_emb-v2-64.pt"
$Users = $Counts[$Dataset].Users
$Items = $Counts[$Dataset].Items

if ([string]::IsNullOrWhiteSpace($Python)) {
    if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    }
    else {
        $Python = "python"
    }
}
Write-Host "Using Python: $Python"

Push-Location $SrcRoot
try {
    if ($Gpu -lt 0) {
        & $Python test_IPG_PreMatch.py `
            --data_dir $DataDir `
            --data_name $Dataset `
            --num_users $Users `
            --num_items $Items `
            --model_idx $ModelIdx `
            --num_intent_clusters $Clusters `
            --batch_size $BatchSize `
            --user_emb_path $UserEmbPath `
            --item_emb_path $ItemEmbPath `
            --device cpu `
            --no_cuda
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    else {
        & $Python test_IPG_PreMatch.py `
            --data_dir $DataDir `
            --data_name $Dataset `
            --num_users $Users `
            --num_items $Items `
            --model_idx $ModelIdx `
            --num_intent_clusters $Clusters `
            --batch_size $BatchSize `
            --user_emb_path $UserEmbPath `
            --item_emb_path $ItemEmbPath `
            --gpu_id $Gpu `
            --device "cuda:$Gpu"
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

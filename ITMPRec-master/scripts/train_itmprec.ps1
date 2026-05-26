param(
    [ValidateSet("ml100k", "lastfm", "steam", "douban_movie")]
    [string]$Dataset = "ml100k",
    [int]$Gpu = 0,
    [int]$ModelIdx = 1,
    [int]$Clusters = 32,
    [int]$Epochs = 300,
    [int]$BatchSize = 256,
    [string]$ItemEmbPath = "",
    [string]$BridgeRegPath = "",
    [double]$BridgeRegWeight = 0.0,
    [int]$BridgeRegSamples = 256,
    [double]$BridgeRegMargin = 0.05,
    [double]$BridgeRegMinScoreGap = 2.0,
    [ValidateSet("rank_all", "extreme", "top_bottom")]
    [string]$BridgeRegPairMode = "rank_all",
    [double]$BridgeRegPosTransitionMin = 3.0,
    [double]$BridgeRegPosRiskMax = 3.0,
    [double]$BridgeRegNegTransitionMax = 2.0,
    [double]$BridgeRegNegRiskMin = 4.0,
    [int]$BridgeRegTopK = 8,
    [int]$BridgeRegBottomK = 12,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SrcRoot = Join-Path $RepoRoot "src"
$DataDir = "..\data\$Dataset\"
if ([string]::IsNullOrWhiteSpace($ItemEmbPath)) {
    $ItemEmbPath = "..\data\$Dataset\item_emb-v2-64.pt"
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    }
    else {
        $Python = "python"
    }
}
Write-Host "Using Python: $Python"

$ExtraArgs = @()
if (-not [string]::IsNullOrWhiteSpace($BridgeRegPath)) {
    $ExtraArgs += @("--bridge_reg_path", $BridgeRegPath)
    $ExtraArgs += @("--bridge_reg_weight", $BridgeRegWeight)
    $ExtraArgs += @("--bridge_reg_samples", $BridgeRegSamples)
    $ExtraArgs += @("--bridge_reg_margin", $BridgeRegMargin)
    $ExtraArgs += @("--bridge_reg_min_score_gap", $BridgeRegMinScoreGap)
    $ExtraArgs += @("--bridge_reg_pair_mode", $BridgeRegPairMode)
    $ExtraArgs += @("--bridge_reg_pos_transition_min", $BridgeRegPosTransitionMin)
    $ExtraArgs += @("--bridge_reg_pos_risk_max", $BridgeRegPosRiskMax)
    $ExtraArgs += @("--bridge_reg_neg_transition_max", $BridgeRegNegTransitionMax)
    $ExtraArgs += @("--bridge_reg_neg_risk_min", $BridgeRegNegRiskMin)
    $ExtraArgs += @("--bridge_reg_top_k", $BridgeRegTopK)
    $ExtraArgs += @("--bridge_reg_bottom_k", $BridgeRegBottomK)
}

Push-Location $SrcRoot
try {
    if ($Gpu -lt 0) {
        & $Python main.py `
            --data_dir $DataDir `
            --data_name $Dataset `
            --model_idx $ModelIdx `
            --num_intent_clusters $Clusters `
            --epochs $Epochs `
            --batch_size $BatchSize `
            --item_emb_path $ItemEmbPath `
            --device cpu `
            --no_cuda `
            @ExtraArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    else {
        & $Python main.py `
            --data_dir $DataDir `
            --data_name $Dataset `
            --model_idx $ModelIdx `
            --num_intent_clusters $Clusters `
            --epochs $Epochs `
            --batch_size $BatchSize `
            --item_emb_path $ItemEmbPath `
            --gpu_id $Gpu `
            --device "cuda:$Gpu" `
            @ExtraArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

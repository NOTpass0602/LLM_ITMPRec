param(
    [ValidateSet("ml100k", "lastfm", "steam", "douban_movie")]
    [string]$Dataset = "ml100k",
    [ValidateSet("baseline", "profiles", "cot", "feedback", "replan", "full")]
    [string]$Ablation = "baseline",
    [int]$Gpu = 0,
    [int]$ModelIdx = 1,
    [int]$Clusters = 32,
    [int]$BatchSize = 256,
    [int]$CandidateK = 50,
    [double]$LirAlpha = 0.20,
    [double]$LirBeta = 0.20,
    [double]$LirGamma = 0.20,
    [double]$LirDelta = 0.30,
    [double]$CotMinTransition = 0.50,
    [double]$CotMaxRisk = 0.50,
    [switch]$CotDisableBonus,
    [switch]$CotUseRiskPenalty,
    [switch]$CotReplanFilter,
    [double]$CotReplanRiskMin = 0.75,
    [double]$CotReplanTransitionMax = 0.25,
    [int]$CotReplanMinRejects = 1,
    [double]$CotReplanMargin = 0.0,
    [switch]$CotReplanProactive,
    [string]$DumpBridgePairsPath = "",
    [int]$DumpPairsPerTarget = 50,
    [string]$ProfilePath = "",
    [string]$CotCachePath = "",
    [string]$ProfileEmbeddingPath = "",
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
$LirProfilePath = if ([string]::IsNullOrWhiteSpace($ProfilePath)) { "" } else { $ProfilePath }
$LirCotCachePath = if ([string]::IsNullOrWhiteSpace($CotCachePath)) { "" } else { $CotCachePath }
$LirProfileEmbeddingPath = if ([string]::IsNullOrWhiteSpace($ProfileEmbeddingPath)) { "" } else { $ProfileEmbeddingPath }
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
    $ExtraArgs = @()
    if (-not [string]::IsNullOrWhiteSpace($LirProfilePath)) {
        $ExtraArgs += @("--lir_profile_path", $LirProfilePath)
    }
    if (-not [string]::IsNullOrWhiteSpace($LirCotCachePath)) {
        $ExtraArgs += @("--lir_cot_cache_path", $LirCotCachePath)
    }
    if (-not [string]::IsNullOrWhiteSpace($LirProfileEmbeddingPath)) {
        $ExtraArgs += @("--lir_profile_embedding_path", $LirProfileEmbeddingPath)
    }
    if (-not [string]::IsNullOrWhiteSpace($DumpBridgePairsPath)) {
        $ExtraArgs += @("--lir_dump_bridge_pairs_path", $DumpBridgePairsPath)
        $ExtraArgs += @("--lir_dump_pairs_per_target", $DumpPairsPerTarget)
    }
    $ExtraArgs += @("--lir_cot_min_transition", $CotMinTransition)
    $ExtraArgs += @("--lir_cot_max_risk", $CotMaxRisk)
    if ($CotDisableBonus) {
        $ExtraArgs += @("--lir_disable_cot_bonus")
    }
    if ($CotUseRiskPenalty) {
        $ExtraArgs += @("--lir_cot_use_risk_penalty")
    }
    if ($CotReplanFilter) {
        $ExtraArgs += @("--lir_cot_replan_filter")
        $ExtraArgs += @("--lir_cot_replan_risk_min", $CotReplanRiskMin)
        $ExtraArgs += @("--lir_cot_replan_transition_max", $CotReplanTransitionMax)
        $ExtraArgs += @("--lir_cot_replan_min_rejects", $CotReplanMinRejects)
        $ExtraArgs += @("--lir_cot_replan_margin", $CotReplanMargin)
        if ($CotReplanProactive) {
            $ExtraArgs += @("--lir_cot_replan_proactive")
        }
    }

    if ($Gpu -lt 0) {
        & $Python test_LIR_PreMatch.py `
            --data_dir $DataDir `
            --data_name $Dataset `
            --num_users $Users `
            --num_items $Items `
            --model_idx $ModelIdx `
            --num_intent_clusters $Clusters `
            --batch_size $BatchSize `
            --user_emb_path $UserEmbPath `
            --item_emb_path $ItemEmbPath `
            --lir_ablation $Ablation `
            --lir_candidate_k $CandidateK `
            --lir_alpha $LirAlpha `
            --lir_beta $LirBeta `
            --lir_gamma $LirGamma `
            --lir_delta $LirDelta `
            --device cpu `
            --no_cuda `
            @ExtraArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    else {
        & $Python test_LIR_PreMatch.py `
            --data_dir $DataDir `
            --data_name $Dataset `
            --num_users $Users `
            --num_items $Items `
            --model_idx $ModelIdx `
            --num_intent_clusters $Clusters `
            --batch_size $BatchSize `
            --user_emb_path $UserEmbPath `
            --item_emb_path $ItemEmbPath `
            --lir_ablation $Ablation `
            --lir_candidate_k $CandidateK `
            --lir_alpha $LirAlpha `
            --lir_beta $LirBeta `
            --lir_gamma $LirGamma `
            --lir_delta $LirDelta `
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

param(
    [ValidateSet("ml100k")]
    [string]$Dataset = "ml100k",
    [string]$Model = "qwen2.5:7b-instruct",
    [string]$OllamaUrl = "http://localhost:11434",
    [string]$PairPath = "",
    [string]$ProfilePath = "",
    [string]$CotPath = "",
    [int]$MaxPairs = 0,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    }
    else {
        $Python = "python"
    }
}

$DataDir = Join-Path $RepoRoot "data\$Dataset"
if ([string]::IsNullOrWhiteSpace($PairPath)) {
    $PairPath = Join-Path $DataDir "lir_bridge_pairs_top50.jsonl"
}
if ([string]::IsNullOrWhiteSpace($ProfilePath)) {
    $ProfilePath = Join-Path $DataDir "lir_profiles_ollama.jsonl"
}
if ([string]::IsNullOrWhiteSpace($CotPath)) {
    $CotPath = Join-Path $DataDir "cot_bridge_cache_ollama_pair_top50.jsonl"
}

Write-Host "Using Python: $Python"
Write-Host "Ollama model: $Model"
Write-Host "Pair path: $PairPath"
Write-Host "Profile path: $ProfilePath"
Write-Host "Cot path: $CotPath"

& $Python (Join-Path $RepoRoot "tools\generate_ml100k_lir_files_ollama.py") `
    --mode pair_cot `
    --data_dir $DataDir `
    --raw_root (Join-Path (Split-Path -Parent $RepoRoot) "ml-100k\ml-100k") `
    --profile_path $ProfilePath `
    --cot_path $CotPath `
    --pair_path $PairPath `
    --model $Model `
    --ollama_url $OllamaUrl `
    --max_pairs $MaxPairs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

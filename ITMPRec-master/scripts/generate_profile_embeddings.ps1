param(
    [ValidateSet("ml100k", "lastfm", "steam", "douban_movie")]
    [string]$Dataset = "ml100k",
    [ValidateSet("rules", "ollama")]
    [string]$ProfileSource = "ollama",
    [string]$ModelName = "BAAI/bge-small-en-v1.5",
    [int]$BatchSize = 64,
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

$DataRoot = Join-Path $RepoRoot "data\$Dataset"
$ProfileFile = if ($ProfileSource -eq "ollama") { "lir_profiles_ollama.jsonl" } else { "lir_profiles.jsonl" }
$ModelLabel = if (Test-Path $ModelName) { Split-Path -Leaf $ModelName } else { $ModelName }
$SafeModelName = $ModelLabel.Replace("/", "_").Replace("\", "_").Replace(":", "_")
$OutputFile = "profile_semantic_emb_${ProfileSource}_${SafeModelName}.pt"

$ProfilePath = Join-Path $DataRoot $ProfileFile
$OutputPath = Join-Path $DataRoot $OutputFile

Write-Host "Using Python: $Python"
Write-Host "Profile path: $ProfilePath"
Write-Host "Output path: $OutputPath"

& $Python (Join-Path $RepoRoot "tools\generate_profile_semantic_embeddings.py") `
    --profile_path $ProfilePath `
    --output_path $OutputPath `
    --model_name $ModelName `
    --batch_size $BatchSize

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

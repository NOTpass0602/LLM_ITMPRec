param(
    [ValidateSet("ml100k")]
    [string]$Dataset = "ml100k",
    [double]$SemanticWeight = 0.20,
    [string]$GraphItemEmb = "",
    [string]$SemanticProfileEmb = "",
    [string]$OutputPath = "",
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
if ([string]::IsNullOrWhiteSpace($GraphItemEmb)) {
    $GraphItemEmb = Join-Path $DataDir "item_emb-v2-64.pt"
}
if ([string]::IsNullOrWhiteSpace($SemanticProfileEmb)) {
    $SemanticProfileEmb = Join-Path $DataDir "profile_semantic_emb_ollama_bge-small-en-v1.5.pt"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $WeightLabel = [int]($SemanticWeight * 100)
    $OutputPath = Join-Path $DataDir "item_emb-llm-fused-w$WeightLabel-64.pt"
}
$MetadataPath = "$OutputPath.json"

Write-Host "Using Python: $Python"
Write-Host "Graph item embedding: $GraphItemEmb"
Write-Host "Semantic profile embedding: $SemanticProfileEmb"
Write-Host "Output path: $OutputPath"

& $Python (Join-Path $RepoRoot "tools\fuse_item_embeddings.py") `
    --graph_item_emb $GraphItemEmb `
    --semantic_profile_emb $SemanticProfileEmb `
    --output_path $OutputPath `
    --semantic_weight $SemanticWeight `
    --metadata_path $MetadataPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

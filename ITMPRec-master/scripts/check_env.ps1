param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:DGLBACKEND = "pytorch"

$Code = @'
import importlib.util
import sys

mods = [
    "torch",
    "dgl",
    "numpy",
    "scipy",
    "tqdm",
    "faiss",
    "kmeans_pytorch",
    "torchdata",
    "fsspec",
]

print(sys.version)
missing = []
for mod in mods:
    ok = importlib.util.find_spec(mod) is not None
    print(f"{mod:20} {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(mod)

if importlib.util.find_spec("tqdm.auto") is None:
    print(f"{'tqdm.auto':20} MISSING")
    missing.append("tqdm.auto")
else:
    print(f"{'tqdm.auto':20} OK")

if importlib.util.find_spec("torch") is not None:
    import torch
    print(f"torch version        {torch.__version__}")
    print(f"cuda available       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda version         {torch.version.cuda}")
        print(f"gpu count            {torch.cuda.device_count()}")

if missing:
    raise SystemExit("Missing modules: " + ", ".join(missing))
'@

$Code | & $Python -
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

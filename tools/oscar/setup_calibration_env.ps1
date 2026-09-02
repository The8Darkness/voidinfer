[CmdletBinding()]
param(
    [string]$PythonExe = 'C:\Users\Micha\AppData\Local\Programs\Python\Python312\python.exe',
    [string]$EnvironmentRoot = 'D:\AI\tools\oscar-calibration',
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

$environmentRootPath = [System.IO.Path]::GetFullPath($EnvironmentRoot)
$venvPath = Join-Path $environmentRootPath '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$verificationPath = Join-Path $environmentRootPath 'verification'

New-Item -ItemType Directory -Force -Path $environmentRootPath | Out-Null
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $PythonExe -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed with exit code $LASTEXITCODE" }
}

if (-not $SkipInstall) {
    & $venvPython -m pip install --disable-pip-version-check `
        --index-url https://download.pytorch.org/whl/cpu `
        'torch==2.13.0+cpu'
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path $verificationPath | Out-Null
$freezePath = Join-Path $environmentRootPath 'requirements.freeze.txt'
& $venvPython -m pip freeze | Set-Content -LiteralPath $freezePath -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed with exit code $LASTEXITCODE" }

$verificationScript = @'
import json
import pathlib
import sys

import torch

root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
tensor = torch.arange(24, dtype=torch.float64).reshape(4, 6)
roundtrip_path = root / "roundtrip.pt"
torch.save(tensor, roundtrip_path)
loaded = torch.load(roundtrip_path, map_location="cpu", weights_only=True)
if not torch.equal(tensor, loaded):
    raise RuntimeError("torch .pt save/load round trip changed tensor contents")

torch.manual_seed(20260901)
sample = torch.randn(128, 128, dtype=torch.float64)
matrix = (sample + sample.T) / 2.0
eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
if eigenvalues.shape != (128,) or eigenvectors.shape != (128, 128):
    raise RuntimeError("unexpected torch.linalg.eigh output shape")
if not torch.isfinite(eigenvalues).all() or not torch.isfinite(eigenvectors).all():
    raise RuntimeError("torch.linalg.eigh produced a non-finite result")

report = {
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda_available": torch.cuda.is_available(),
    "roundtrip_path": str(roundtrip_path),
    "roundtrip_shape": list(loaded.shape),
    "eigh_matrix_shape": list(matrix.shape),
    "eigh_eigenvalues_shape": list(eigenvalues.shape),
    "eigh_eigenvectors_shape": list(eigenvectors.shape),
    "eigh_min_eigenvalue": float(eigenvalues[0]),
    "eigh_max_eigenvalue": float(eigenvalues[-1]),
}
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
'@

$temporaryVerificationScript = Join-Path $verificationPath 'verify_environment.py'
Set-Content -LiteralPath $temporaryVerificationScript -Value $verificationScript -Encoding utf8
try {
    & $venvPython $temporaryVerificationScript $verificationPath
    if ($LASTEXITCODE -ne 0) { throw "calibration environment verification failed with exit code $LASTEXITCODE" }
} finally {
    Remove-Item -LiteralPath $temporaryVerificationScript -Force
}

Write-Output "Environment: $environmentRootPath"
Write-Output "Python: $venvPython"
Write-Output "Freeze: $freezePath"

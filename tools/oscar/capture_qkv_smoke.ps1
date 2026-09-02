[CmdletBinding()]
param(
    [string]$ArtifactPath = 'C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer',
    [string]$BuildRoot = 'D:\AI\build-adaptive-dflash2',
    [string]$OutputDirectory = '',
    [string]$PythonExe = 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$captureExe = Join-Path $BuildRoot 'tests\ninfer_qwen3_6_27b_oscar_capture_test.exe'
$validator = Join-Path $repoRoot 'tools\oscar\validate_dump.py'
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot 'results\oscar\captures\phase-b2-qkv-256'
}
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$relativeOutput = [System.IO.Path]::GetRelativePath($repoRoot, $outputRoot)
if ($relativeOutput.StartsWith('..') -or [System.IO.Path]::IsPathRooted($relativeOutput)) {
    throw "Capture output must remain under the repository: $outputRoot"
}
if (-not (Test-Path -LiteralPath $ArtifactPath -PathType Leaf)) {
    throw "Qwen3.8 artifact not found: $ArtifactPath"
}
if (-not (Test-Path -LiteralPath $captureExe -PathType Leaf)) {
    throw "Capture executable not found; build the focused target first: $captureExe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Capture validator Python not found: $PythonExe"
}
if (Test-Path -LiteralPath $outputRoot) {
    throw "Capture output already exists; choose a fresh output directory: $outputRoot"
}

$sourceFiles = @(
    'src\targets\qwen3_6\impl\runtime\text_context_impl.h',
    'src\targets\qwen3_6\impl\runtime\oscar_qkv_capture.h',
    'src\targets\qwen3_6\impl\runtime\oscar_qkv_capture.cpp',
    'tests\targets\qwen3_6_27b\test_oscar_capture.cpp',
    'tools\oscar\capture_qkv_smoke.ps1'
)
$sourceIdentityParts = foreach ($relativePath in $sourceFiles) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Capture source identity file is missing: $sourcePath"
    }
    "$relativePath=$((Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash.ToLowerInvariant())"
}
$sourceIdentity = $sourceIdentityParts -join ';'
$modelSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArtifactPath).Hash.ToLowerInvariant()
$executableSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $captureExe).Hash.ToLowerInvariant()

$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS = [System.IO.Path]::GetFullPath($ArtifactPath)
$env:NINFER_OSCAR_QKV_CAPTURE_DIR = $outputRoot
$env:NINFER_OSCAR_QKV_MODEL_ID = 'qwen3.8-27b'
$env:NINFER_OSCAR_QKV_WEIGHTS_ID = 'nvfp4-dflash2'
$env:NINFER_OSCAR_QKV_MODEL_SHA256 = $modelSha256
$env:NINFER_OSCAR_QKV_EXECUTABLE_SHA256 = $executableSha256
$env:NINFER_OSCAR_QKV_SOURCE_ID = $sourceIdentity
$env:NINFER_OSCAR_QKV_EXPECTED_TOKENS = '256'

Set-Location $repoRoot
Write-Output "Capture executable: $captureExe"
Write-Output "Artifact: $ArtifactPath"
Write-Output "Output: $outputRoot"
Write-Output "Useful tokens: 256"
& $captureExe
if ($LASTEXITCODE -ne 0) {
    throw "Native OSCAR QKV smoke capture failed with exit code $LASTEXITCODE"
}

$manifestPath = Join-Path $outputRoot 'manifest.json'
$manifestSidecar = Join-Path $outputRoot 'manifest.sha256'
& $PythonExe $validator $manifestPath
if ($LASTEXITCODE -ne 0) {
    throw "OSCAR QKV validation failed before manifest sidecar creation"
}
$manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
Set-Content -LiteralPath $manifestSidecar -Value "$manifestSha256  manifest.json" -Encoding ascii
& $PythonExe $validator $manifestPath --manifest-sha256 $manifestSidecar
if ($LASTEXITCODE -ne 0) {
    throw "OSCAR QKV validation failed with manifest sidecar"
}
Write-Output "Manifest SHA-256: $manifestSha256"
Write-Output "Manifest: $manifestPath"
Write-Output "Manifest sidecar: $manifestSidecar"

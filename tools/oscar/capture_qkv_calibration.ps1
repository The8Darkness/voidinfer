[CmdletBinding()]
param(
    [string]$ArtifactPath = 'C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer',
    [string]$BuildRoot = 'D:\AI\build-adaptive-dflash2',
    [string]$OutputDirectory = '',
    [string]$PythonExe = 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe',
    [int]$Requests = 40,
    [int]$PromptTokens = 256,
    [int]$FormulaSeed = 17,
    [string]$InputDescription = 'deterministic_request_formula_v1'
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$captureExe = Join-Path $BuildRoot 'tests\ninfer_qwen3_6_27b_oscar_capture_test.exe'
$validator = Join-Path $repoRoot 'tools\oscar\validate_calibration_dump.py'
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot 'results\oscar\captures\phase-c3-cal10k'
}
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$relativeOutput = [System.IO.Path]::GetRelativePath($repoRoot, $outputRoot)
$usefulTokens = $Requests * $PromptTokens
if ($Requests -le 0 -or $PromptTokens -le 0 -or $PromptTokens -gt 512 -or $usefulTokens -le 0) {
    throw "Requests and prompt tokens must be positive, prompt tokens <= 512"
}
if ($FormulaSeed -lt 0) {
    throw "FormulaSeed must be nonnegative"
}
if ($relativeOutput.StartsWith('..') -or [System.IO.Path]::IsPathRooted($relativeOutput)) {
    throw "Capture output must remain under the repository: $outputRoot"
}
foreach ($requiredPath in @($ArtifactPath, $captureExe, $PythonExe, $validator)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath"
    }
}
if (Test-Path -LiteralPath $outputRoot) {
    throw "Capture output already exists; choose a fresh output directory: $outputRoot"
}

$sourceFiles = @(
    'src\targets\qwen3_6\impl\runtime\text_context_impl.h',
    'src\targets\qwen3_6\impl\frontend\frontend.cpp',
    'src\targets\qwen3_6\impl\runtime\oscar_qkv_capture.h',
    'src\targets\qwen3_6\impl\runtime\oscar_qkv_capture.cpp',
    'tests\targets\qwen3_6_27b\test_oscar_capture.cpp',
    'tools\oscar\capture_qkv_calibration.ps1',
    'tools\oscar\validate_calibration_dump.py'
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
$env:NINFER_OSCAR_QKV_EXPECTED_TOKENS = $usefulTokens.ToString()
$env:NINFER_OSCAR_QKV_CAPTURE_REQUESTS = $Requests.ToString()
$env:NINFER_OSCAR_QKV_PROMPT_TOKENS = $PromptTokens.ToString()
$env:NINFER_OSCAR_QKV_TOKEN_FORMULA_SEED = $FormulaSeed.ToString()
$env:NINFER_OSCAR_QKV_INPUT_DESCRIPTION = $InputDescription
$env:NINFER_OSCAR_QKV_CAPTURE_ONLY = '1'

Set-Location $repoRoot
Write-Output "Capture executable: $captureExe"
Write-Output "Artifact: $ArtifactPath"
Write-Output "Output: $outputRoot"
Write-Output "Requests: $Requests prompt_tokens: $PromptTokens useful_tokens: $usefulTokens"
Write-Output "Input description: $InputDescription formula_seed: $FormulaSeed"
& $captureExe
if ($LASTEXITCODE -ne 0) {
    throw "Native OSCAR calibration capture failed with exit code $LASTEXITCODE"
}

$manifestPath = Join-Path $outputRoot 'manifest.json'
$manifestSidecar = Join-Path $outputRoot 'manifest.sha256'
$manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
Set-Content -LiteralPath $manifestSidecar -Value "$manifestSha256  manifest.json" -Encoding ascii
& $PythonExe $validator $manifestPath --manifest-sha256 $manifestSidecar --expected-useful-tokens $usefulTokens --expected-chunks $Requests
if ($LASTEXITCODE -ne 0) {
    throw "OSCAR calibration validation failed"
}
Write-Output "Manifest SHA-256: $manifestSha256"
Write-Output "Manifest: $manifestPath"
Write-Output "Manifest sidecar: $manifestSidecar"

<#
.SYNOPSIS
Builds the prebuilt Windows package (voidinfer-<version>-win64-sm120a.zip) from a Release build.

.DESCRIPTION
Run from a VS 2026 developer shell (VsDevCmd.bat -arch=amd64), so dumpbin and VCToolsRedistDir are
available. The package holds voidinfer.exe and voidinfer-serve.exe with every non-system DLL they
load, resolved by walking dumpbin /dependents from the build output, the CUDA 13.3 bin directory
(cuBLAS, cuBLASLt) and the app-local MSVC runtime, so it runs without CUDA, vcpkg or the VC++
redistributable installed. It also carries the target's OSCAR rotations (l0_oscar/, not published
with the Hugging Face weights) and the license notices of everything it ships.
#>
param(
    [Parameter(Mandatory)] [string] $BuildDir,
    [Parameter(Mandatory)] [string] $RotationsDir,
    [string] $OutDir = (Join-Path $PSScriptRoot '..\..\dist'),
    [string] $Version
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$apps = (Resolve-Path (Join-Path $BuildDir 'apps')).Path
if (-not $Version) { $Version = "$(Get-Date -Format yyyy.MM.dd)-$((git -C $repo rev-parse --short HEAD).Trim())" }
if (-not (Get-Command dumpbin -ErrorAction SilentlyContinue) -or -not $env:VCToolsRedistDir) {
    throw 'Run from a VS 2026 developer shell (VsDevCmd.bat -arch=amd64).'
}
$cudaRoot = $env:CUDA_PATH_V13_3
if (-not $cudaRoot) { throw 'CUDA_PATH_V13_3 is not set.' }
$crt = Join-Path $env:VCToolsRedistDir 'x64\Microsoft.VC145.CRT'
$search = @($apps, (Join-Path $cudaRoot 'bin\x64'), (Join-Path $cudaRoot 'bin'), $crt) | Where-Object { Test-Path $_ }
foreach ($f in 'k_mean.bin', 'k_rot_perhead.bin', 'v_rot_perhead.bin') {
    if (-not (Test-Path (Join-Path $RotationsDir $f))) { throw "$RotationsDir has no $f" }
}

$name = "voidinfer-$Version-win64-sm120a"
$stage = Join-Path $OutDir $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage, (Join-Path $stage 'licenses'), (Join-Path $stage 'l0_oscar') | Out-Null

# Dependency closure: a DLL found in a search directory is shipped; anything else must be a system DLL.
$shipped = @{}
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($exe in 'voidinfer.exe', 'voidinfer-serve.exe') {
    $path = Join-Path $apps $exe
    if (-not (Test-Path $path)) { throw "$path is missing; build the voidinfer and voidinfer-serve targets." }
    Copy-Item $path $stage
    $queue.Enqueue($path)
}
while ($queue.Count) {
    $deps = dumpbin /nologo /dependents $queue.Dequeue() | Where-Object { $_ -match '^\s+(\S+\.dll)\s*$' } | ForEach-Object { $Matches[1] }
    foreach ($dll in $deps) {
        $key = $dll.ToLowerInvariant()
        if ($shipped.ContainsKey($key) -or $key.StartsWith('api-ms-') -or $key.StartsWith('ext-ms-')) { continue }
        $hit = $search | ForEach-Object { Join-Path $_ $dll } | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($hit) {
            $shipped[$key] = $hit
            $queue.Enqueue($hit)
        } elseif (-not (Test-Path (Join-Path $env:SystemRoot "System32\$dll"))) {
            throw "Unresolved dependency $dll"
        }
    }
}
$shipped.Values | ForEach-Object { Copy-Item $_ $stage }
Copy-Item (Join-Path $RotationsDir '*.bin') (Join-Path $stage 'l0_oscar')

# License notices: VoidInfer, vendored sources, vcpkg ports (FFmpeg is the LGPL build), CUDA, MSVC runtime.
$lic = Join-Path $stage 'licenses'
Copy-Item (Join-Path $repo 'LICENSE') (Join-Path $lic 'VoidInfer-LICENSE.txt')
Copy-Item (Join-Path $repo 'UPSTREAM_AUDIT.md') $lic
Copy-Item (Join-Path $repo 'src\exl3\mia_exllamav3\LICENSE') (Join-Path $lic 'exllamav3-LICENSE.txt')
Copy-Item (Join-Path $repo 'third_party\cpp-httplib\LICENSE') (Join-Path $lic 'cpp-httplib-LICENSE.txt')
Copy-Item (Join-Path $repo 'third_party\nlohmann\LICENSE.MIT') (Join-Path $lic 'nlohmann-json-LICENSE.txt')
Copy-Item (Join-Path $repo 'third_party\utf8proc\LICENSE.md') (Join-Path $lic 'utf8proc-LICENSE.md')
$cache = Get-Content (Join-Path $BuildDir 'CMakeCache.txt')
$ffInclude = ($cache | Where-Object { $_ -match '^FFMPEG_INCLUDE_DIRS:\w+=(.+)$' } | ForEach-Object { $Matches[1] }) -split ';' | Select-Object -First 1
$share = Join-Path (Split-Path $ffInclude -Parent) 'share'
foreach ($port in 'ffmpeg', 'curl', 'zlib') {
    Copy-Item (Join-Path $share "$port\copyright") (Join-Path $lic "$port-copyright.txt")
}
Copy-Item (Join-Path $cudaRoot 'LICENSE') (Join-Path $lic 'NVIDIA-CUDA-EULA.txt')
$vsLicense = Get-ChildItem (Join-Path $env:VSINSTALLDIR 'Licenses') -Recurse -Filter '*.rtf' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($vsLicense) { Copy-Item $vsLicense.FullName (Join-Path $lic 'MSVC-runtime-license.rtf') }

@"
VoidInfer $Version - prebuilt for Windows 11 x64 and the RTX 5090 (sm_120a)

Setup and usage: https://github.com/The8Darkness/voidinfer/blob/main/docs/build-and-run.md#prebuilt-windows-package

1. Download the target (Mia-AiLab/Qwen3.8-27B-EXL3, branch SC_6.00bpw_H6_V6) and the draft
   (Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) from Hugging Face.
2. Copy the l0_oscar folder of this package into the target directory.
3. voidinfer-serve.exe --target <target dir> --draft <draft dir> --model-id qwen3.8-27b

Shipped DLLs: FFmpeg (LGPL 2.1, vcpkg build), libcurl, zlib, NVIDIA cuBLAS/cuBLASLt (CUDA EULA
redistributable), Microsoft Visual C++ runtime. Notices are in licenses\.
"@ | Set-Content -Encoding utf8 (Join-Path $stage 'README.txt')

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
"$hash  $name.zip" | Set-Content -Encoding ascii "$zip.sha256"
Write-Host "Shipped DLLs: $(($shipped.Keys | Sort-Object) -join ', ')"
Write-Host "$zip ($([math]::Round((Get-Item $zip).Length / 1MB)) MB) sha256 $hash"

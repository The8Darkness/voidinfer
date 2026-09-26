param(
    [string] $BuildToolsRoot = 'C:\BuildTools',
    [string] $MsvcVersion = '14.44.35207',
    [string] $WindowsSdkRoot = 'C:\Program Files (x86)\Windows Kits\10',
    [string] $WindowsSdkVersion = '10.0.26100.0'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-CleanupJob.ps1')
$sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$output = Join-Path $sourceRoot 'build-host-contracts'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$msvc = Join-Path $BuildToolsRoot "VC\Tools\MSVC\$MsvcVersion"
$compilerDirectory = Join-Path $msvc 'bin\Hostx64\x64'
$compiler = Join-Path $compilerDirectory 'cl.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw "Missing compiler: $compiler" }
# Direct host compilation avoids the CUDA project's configure step and all
# developer-shell, package-manager, discovery and post-build hooks.
$env:PATH = "$compilerDirectory;$env:SystemRoot\System32"
$env:INCLUDE = "$msvc\include;$WindowsSdkRoot\Include\$WindowsSdkVersion\ucrt;$WindowsSdkRoot\Include\$WindowsSdkVersion\shared;$WindowsSdkRoot\Include\$WindowsSdkVersion\um"
$env:LIB = "$msvc\lib\x64;$WindowsSdkRoot\Lib\$WindowsSdkVersion\ucrt\x64;$WindowsSdkRoot\Lib\$WindowsSdkVersion\um\x64"
$env:CL = ''
$env:_CL_ = ''
$env:LINK = ''
$env:_LINK_ = ''
# Closed allowlist: these sources and their transitive project headers contain
# only standard-library host contracts. No ninfer_core or CUDA linkage.
foreach ($test in @('test_exl3_environment_options', 'test_exl3_candidate_contracts')) {
    $executable = Join-Path $output "$test.exe"
    & $compiler /nologo /std:c++20 /EHsc /MD /Od /MP1 /cgthreads1 /W4 `
        /D_CRT_SECURE_NO_WARNINGS "/I$sourceRoot\src" "/I$sourceRoot\tests" `
        (Join-Path $sourceRoot "tests\$test.cpp") `
        "/Fo$output\$test.obj" "/Fe$executable" /link /INCREMENTAL:NO
    if ($LASTEXITCODE -ne 0) { throw "$test compilation failed: $LASTEXITCODE" }
    & $executable
    if ($LASTEXITCODE -ne 0) { throw "$test failed: $LASTEXITCODE" }
}
Write-Output 'PASS: two audited EXL3 host contracts; no CUDA build or execution'

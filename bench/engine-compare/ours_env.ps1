# Shared environment for our side of the harness (dot-source it). The engine applies its benchmarked
# profile itself (src/exl3/engine_profile.cpp), so a run only needs the model package, the build and
# per-run VOIDINFER_* overrides. Receipt CSVs written before the rename use NINFER_* names; they are
# mapped to VOIDINFER_* when a test route still needs one (ours_bench.ps1 base mode).
$VoidBuild  = 'D:\AI\voidinfer-sol6-campaign-20260923\builds\B2'
$VoidTarget = 'C:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6'      # l0_oscar/ holds the OSCAR rotations
$VoidDraft  = 'C:\AI\models\Qwen3.8-27B-DFlash2-EXL3-5bpw'
$VoidVision = 'D:\AI\models\Qwen3.8-27B-bf16'
$VoidServe  = Join-Path $VoidBuild 'apps\voidinfer-serve.exe'
$env:PATH = 'D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;' + $env:PATH

function Reset-VoidEnv([string]$Overrides = '', [string]$Csv = '') {
    # Remove-Item deletes; [Environment]::SetEnvironmentVariable(name, $null) would leave an empty value.
    Get-ChildItem Env: | Where-Object { $_.Name -like 'VOIDINFER*' -or $_.Name -like 'NINFER*' } |
        ForEach-Object { Remove-Item -LiteralPath ('Env:' + $_.Name) }
    if ($Csv) {
        foreach ($row in (Import-Csv $Csv)) {
            $name = $row.Name -replace '^NINFER_', 'VOIDINFER_'
            if ($name -notmatch 'L0_OSCAR') { [Environment]::SetEnvironmentVariable($name, $row.Value, 'Process') }
        }
    }
    # Test executables and research tools take the model package from the environment.
    $env:VOIDINFER_EXL3_TARGET_PATH  = $VoidTarget
    $env:VOIDINFER_EXL3_DFLASH2_PATH = $VoidDraft
    foreach ($kv in ($Overrides -split ';' | Where-Object { $_ })) {
        $k, $v = $kv -split '=', 2
        [Environment]::SetEnvironmentVariable($k, $v, 'Process')
    }
}

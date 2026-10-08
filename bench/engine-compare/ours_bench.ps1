param(
    [Parameter(Mandatory=$true)][string[]]$Prompts,   # name:prefix, e.g. code_4k:4096
    [ValidateSet('dflash2','base')][string]$Mode='dflash2',
    [int]$Outputs=1024,
    [string]$Out='D:\AI\engine-compare-20261008\results\ours',
    [string]$Build='D:\AI\voidinfer-sol6-campaign-20260923\builds\B2',
    [string]$Tag='',
    [string]$Env='',        # extra NAME=VALUE;... applied after the receipt environment
    [switch]$L0,             # force L0 OSCAR even at <=32K
    [string]$NsysOut=''      # optional: trace CUDA allocations into this report path prefix
)
# Our side of the 1:1 comparison. DFlash2: public Engine (device harness, warm-only: a same-length
# warm request precedes the measured one) with the shared messages; it exports the exact prompt ids.
# Base: target-only TargetRequest route replaying those exported prompt ids.
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force $Out | Out-Null
$runs='D:\AI\voidinfer-sol6-campaign-20260923\runs'
$fast=(Get-ChildItem $runs -Directory | ? { $_.Name -like '*FastDeviceTransaction-B2' -and (Test-Path (Join-Path $_.FullName 'fast-device-environment.csv')) -and -not (Test-Path (Join-Path $_.FullName 'verify-quality.csv')) } | Sort Name | Select -Last 1).FullName
$target=(Get-ChildItem $runs -Directory | ? { $_.Name -like '*TargetRequest-B2' -and (Test-Path (Join-Path $_.FullName 'target-environment.csv')) } | Sort Name | Select -Last 1).FullName
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
function Reset-Env($csv,[switch]$KeepL0) {
    foreach($item in @(Get-ChildItem Env:NINFER* -ErrorAction SilentlyContinue)) { Remove-Item -LiteralPath ("Env:"+$item.Name) }
    foreach($row in (Import-Csv $csv)) { if($KeepL0 -or $row.Name -notmatch 'L0_OSCAR'){ [Environment]::SetEnvironmentVariable($row.Name,$row.Value,'Process') } }
}
foreach($case in $Prompts){
    $name,$prefix=$case -split ':'; $prefix=[int]$prefix
    $stem=Join-Path $Out "$name.$Mode$Tag"
    $pm="D:\AI\engine-compare-20261008\prompts\$name.pmsg"
    $ids="$Out\$name.prompt.ids"
    if($Mode -eq 'dflash2'){
        $l0 = [bool]$L0 -or ($prefix -gt 32768)
        Reset-Env "$fast\fast-device-environment.csv" -KeepL0:$l0
        if($l0){ $env:NINFER_EXL3_L0_OSCAR='1'; $env:NINFER_EXL3_L0_OSCAR_ROT='D:\AI\kvtier\rot_center384k' }
        foreach($kv in ($Env -split ';' | ? {$_})){ $k,$v=$kv -split '=',2; [Environment]::SetEnvironmentVariable($k,$v,'Process') }
        $env:NINFER_TEST_ENGINE_DEVICE_PREFIX=[string]$prefix
        $env:NINFER_TEST_ENGINE_DEVICE_SOURCE='D:\AI\kvtier\long_code_source.cpp'
        $env:NINFER_TEST_ENGINE_DEVICE_FIXTURE='code'
        $env:NINFER_TEST_ENGINE_DEVICE_LONG='1'
        $env:NINFER_TEST_ENGINE_DEVICE_WARM_ONLY='1'
        $env:NINFER_TEST_ENGINE_DEVICE_OUTPUT_TOKENS=[string]$Outputs
        $env:NINFER_TEST_ENGINE_DEVICE_MESSAGES=$pm
        $env:NINFER_TEST_ENGINE_DEVICE_IDS_OUT=$ids
        $env:NINFER_TEST_ENGINE_DEVICE_OUTPUT_OUT="$stem.txt"
        $env:NINFER_TEST_ENGINE_DEVICE_TOKEN_IDS_OUT="$stem.out.ids"
        $env:NINFER_E5A4_MAXCTX=[string]($prefix+$Outputs+256)
        $env:NINFER_EXL3_TEST_ENGINE_FINAL_STATE_HASH='0'
        $exe="$Build\tests\ninfer_exl3_engine_device_test.exe"
        $log=if($NsysOut){
            & 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.4.1\target-windows-x64\nsys.exe' profile -t cuda --cuda-memory-usage=true --force-overwrite=true -o "$NsysOut-$name" $exe 2>&1 | Select-String -NotMatch "EXL3_GENERIC|WIDE_K6"
        } else { & $exe 2>&1 | Select-String -NotMatch "EXL3_GENERIC|WIDE_K6" }
        $log | Out-File "$stem.log"
        $l=($log | Select-String 'ENGINE_LONG |ENGINE_FAIL' | % Line) -join ' '
        $acc=($log | Select-String 'ENGINE_LONG_ROUNDS' | % Line)
        if($l -match 'prompt=(\d+) output=(\d+).*rounds=(\S+).*prefill_seconds=(\S+).*decode_seconds=(\S+)'){
            $p=[int]$Matches[1];$o=[int]$Matches[2];$rd=[double]$Matches[3];$ps=[double]$Matches[4];$ds=[double]$Matches[5]
            $r=[ordered]@{engine='ours';mode=$Mode;tag=$Tag;env=$Env;prompt=$name;prompt_tokens=$p;prefill_s=$ps;prefill_tps=$p/$ps;out_tokens=$o;decode_s=$ds;decode_tps=($o-1)/$ds;rounds=$rd;tok_per_round=$o/$rd;ms_per_round=$ds*1000/$rd;l0_oscar=$l0}
            $r | ConvertTo-Json -Compress | Tee-Object -Append "$Out\results.jsonl"
        } else { "FAIL $name"; $log | Select -Last 8 }
    } else {
        if(-not (Test-Path $ids)){ throw "run dflash2 first to export $ids" }
        Reset-Env "$target\target-environment.csv"
        $n=(Get-Content $ids | ? {$_}).Count
        $dir="$stem.run"; if(Test-Path $dir){ Remove-Item -Recurse -Force $dir }; New-Item -ItemType Directory $dir | Out-Null
        $env:NINFER_E5A4_PROMPT_FILE=$ids
        $env:NINFER_E5A4_CONTEXTS=[string]$n
        $env:NINFER_E5A4_MAXCTX=[string]($n+$Outputs+64)
        $env:NINFER_TARGETREQUEST_OUTPUTS=[string]($Outputs+1)
        $env:NINFER_E5A4_OUT="$dir\summary.csv"
        $env:NINFER_TARGETREQUEST_TOKENS_OUT="$dir\tokens.csv"
        $env:NINFER_REAL_DFLASH_PREFIX=[string]$prefix
        $log=& "$Build\tests\ninfer_exl3_dflash2_accept_test.exe" 2>&1
        $log | Out-File "$dir\target-request.log"
        if(Test-Path "$dir\summary.csv"){
            $s=Import-Csv "$dir\summary.csv" | Select -First 1
            $pm_=[double]$s.ingestion_wall_ms; $dm=[double]$s.decode_execution_wall_ms
            $r=[ordered]@{engine='ours';mode='base';prompt=$name;prompt_tokens=$n;prefill_s=$pm_/1000;prefill_tps=$n*1000/$pm_;out_tokens=$Outputs+1;decode_s=$dm/1000;decode_tps=$Outputs*1000/$dm}
            $r | ConvertTo-Json -Compress | Tee-Object -Append "$Out\results.jsonl"
        } else { "FAIL base $name"; $log | Select -Last 8 }
    }
}

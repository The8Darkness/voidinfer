param(
    [Parameter(Mandatory=$true)][string[]]$Prompts,   # name:prefix, e.g. code_4k:4096
    [ValidateSet('dflash2','base')][string]$Mode='dflash2',
    [int]$Outputs=1024,
    [string]$Out='D:\AI\engine-compare-20261008\results\ours',
    [string]$Build='',
    [string]$Tag='',
    [string]$Env='',        # extra VOIDINFER_NAME=VALUE;... overrides
    [switch]$L0,             # force L0 OSCAR even at <=32K
    [string]$NsysOut=''      # optional: trace CUDA allocations into this report path prefix
)
# Our side of the 1:1 comparison. DFlash2: public Engine (device harness, warm-only: a same-length
# warm request precedes the measured one) with the shared messages; it exports the exact prompt ids.
# Base: target-only TargetRequest route replaying those exported prompt ids.
$ErrorActionPreference='Stop'
. "$PSScriptRoot\ours_env.ps1"
if(-not $Build){ $Build=$VoidBuild }
New-Item -ItemType Directory -Force $Out | Out-Null
$runs='D:\AI\voidinfer-sol6-campaign-20260923\runs'
$target=(Get-ChildItem $runs -Directory | ? { $_.Name -like '*TargetRequest-B2' -and (Test-Path (Join-Path $_.FullName 'target-environment.csv')) } | Sort Name | Select -Last 1).FullName
foreach($case in $Prompts){
    $name,$prefix=$case -split ':'; $prefix=[int]$prefix
    $stem=Join-Path $Out "$name.$Mode$Tag"
    $pm="D:\AI\engine-compare-20261008\prompts\$name.pmsg"
    $ids="$Out\$name.prompt.ids"
    if($Mode -eq 'dflash2'){
        # L0 OSCAR is the engine default; -L0 / >32K only label the result row.
        $l0 = [bool]$L0 -or ($prefix -gt 32768)
        Reset-VoidEnv $Env
        $env:VOIDINFER_TEST_ENGINE_DEVICE_PREFIX=[string]$prefix
        $env:VOIDINFER_TEST_ENGINE_DEVICE_SOURCE='D:\AI\kvtier\long_code_source.cpp'
        $env:VOIDINFER_TEST_ENGINE_DEVICE_FIXTURE='code'
        $env:VOIDINFER_TEST_ENGINE_DEVICE_LONG='1'
        $env:VOIDINFER_TEST_ENGINE_DEVICE_WARM_ONLY='1'
        $env:VOIDINFER_TEST_ENGINE_DEVICE_OUTPUT_TOKENS=[string]$Outputs
        $env:VOIDINFER_TEST_ENGINE_DEVICE_MESSAGES=$pm
        $env:VOIDINFER_TEST_ENGINE_DEVICE_IDS_OUT=$ids
        $env:VOIDINFER_TEST_ENGINE_DEVICE_OUTPUT_OUT="$stem.txt"
        $env:VOIDINFER_TEST_ENGINE_DEVICE_TOKEN_IDS_OUT="$stem.out.ids"
        $env:VOIDINFER_E5A4_MAXCTX=[string]($prefix+$Outputs+256)
        $env:VOIDINFER_EXL3_TEST_ENGINE_FINAL_STATE_HASH='0'
        $exe="$Build\tests\voidinfer_exl3_engine_device_test.exe"
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
        Reset-VoidEnv $Env "$target\target-environment.csv"
        $n=(Get-Content $ids | ? {$_}).Count
        $dir="$stem.run"; if(Test-Path $dir){ Remove-Item -Recurse -Force $dir }; New-Item -ItemType Directory $dir | Out-Null
        $env:VOIDINFER_E5A4_PROMPT_FILE=$ids
        $env:VOIDINFER_E5A4_CONTEXTS=[string]$n
        $env:VOIDINFER_E5A4_MAXCTX=[string]($n+$Outputs+64)
        $env:VOIDINFER_TARGETREQUEST_OUTPUTS=[string]($Outputs+1)
        $env:VOIDINFER_E5A4_OUT="$dir\summary.csv"
        $env:VOIDINFER_TARGETREQUEST_TOKENS_OUT="$dir\tokens.csv"
        $env:VOIDINFER_REAL_DFLASH_PREFIX=[string]$prefix
        $log=& "$Build\tests\voidinfer_exl3_dflash2_accept_test.exe" 2>&1
        $log | Out-File "$dir\target-request.log"
        if(Test-Path "$dir\summary.csv"){
            $s=Import-Csv "$dir\summary.csv" | Select -First 1
            $pm_=[double]$s.ingestion_wall_ms; $dm=[double]$s.decode_execution_wall_ms
            $r=[ordered]@{engine='ours';mode='base';prompt=$name;prompt_tokens=$n;prefill_s=$pm_/1000;prefill_tps=$n*1000/$pm_;out_tokens=$Outputs+1;decode_s=$dm/1000;decode_tps=$Outputs*1000/$dm}
            $r | ConvertTo-Json -Compress | Tee-Object -Append "$Out\results.jsonl"
        } else { "FAIL base $name"; $log | Select -Last 8 }
    }
}


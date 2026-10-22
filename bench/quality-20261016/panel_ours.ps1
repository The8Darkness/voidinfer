param([string]$Tag='fp16kv',[string]$Env='VOIDINFER_EXL3_L0_OSCAR=0',[string]$Only='',[int]$Context=64,[int]$K=100,
      [string]$Panel='D:\AI\quality-bench-20261016\panel')
# Teacher-forced panel scoring for our engine (research_exl3_long_nll): per stream, the first $Context
# tokens are prefilled (full-precision wide chunks), then every later token is fed one decode row at a
# time (the serving decode route) and the next-token distribution is recorded: top-$K ids + log-probs
# (.topk) and nll/argmax/target/margin (.tok). Positions predict tokens [$Context, N).
# Default arm: EXL3 6bpw + FP16 device KV (VOIDINFER_EXL3_L0_OSCAR=0, reference measurement only).
. 'D:\AI\voidinfer-sol6-high-4000-70-200-20260923\bench\engine-compare\ours_env.ps1'
$out = "D:\AI\quality-bench-20261016\runs\ours_$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
$t0 = Get-Date
foreach ($ids in Get-ChildItem "$Panel\*.ids" | Where-Object { -not $Only -or $_.Name -like "$Only*" }) {
    $id = $ids.BaseName
    if (Test-Path "$out\$id.topk") { continue }
    Reset-VoidEnv $Env
    $n = (Get-Content $ids.FullName -Raw).Trim().Split(' ').Count
    $env:LONG_NLL_IDS = $ids.FullName; $env:LONG_NLL_CONTEXT = [string]$Context
    $env:LONG_NLL_CONTINUATION = [string]($n - $Context); $env:LONG_NLL_OUT = "$out\$id.tok.part"
    $env:LONG_NLL_TOPK = [string]$K; $env:LONG_NLL_TOPK_OUT = "$out\$id.topk.part"
    $s = Get-Date
    $l = & "$VoidBuild\tests\research_exl3_long_nll.exe" 2>&1 | Select-String 'LONG_NLL' | ForEach-Object Line
    if ($LASTEXITCODE -ne 0) { "FAIL $id $l"; continue }
    Move-Item -Force "$out\$id.tok.part" "$out\$id.tok"; Move-Item -Force "$out\$id.topk.part" "$out\$id.topk"
    "$id $l $([int]((Get-Date) - $s).TotalSeconds)s"
}
"PANEL_DONE $Tag $([int]((Get-Date) - $t0).TotalSeconds)s"

param([int[]]$Windows=@(4096,16384,32768),[string]$Build='',[switch]$L0,[string]$Only='',[string]$Env='',[string]$Tag='',[switch]$TopK)
# Paired NLL for our engine: scores tokens [W, W+512) given tokens [512, W) = NInfer window 2 of --context W --stride 512.
# The tool applies the engine profile itself, so it runs the default L0 OSCAR KV tier; -L0 names the
# outputs .ours_l0. An FP16 device-KV reference needs -Env 'VOIDINFER_EXL3_L0_OSCAR=0'.
. "$PSScriptRoot\..\ours_env.ps1"
if(-not $Build){ $Build=$VoidBuild }
$q='D:\AI\engine-compare-20261008\quality'
foreach($W in $Windows){
  foreach($ids in Get-ChildItem "$q\w$W\*.tail.ids" | ? { -not $Only -or $_.Name -like "$Only*" }){
    Reset-VoidEnv $Env
    $id=$ids.Name -replace '\.tail\.ids$',''
    $out=if($L0){"$q\w$W\$id.ours_l0$Tag.tok"}else{"$q\w$W\$id.ours$Tag.tok"}
    if($TopK){ $env:LONG_NLL_TOPK_OUT=($out -replace '\.tok$','.topk') } else { Remove-Item Env:LONG_NLL_TOPK_OUT -ErrorAction SilentlyContinue }; $env:LONG_NLL_IDS=$ids.FullName; $env:LONG_NLL_CONTEXT=[string]($W-512); $env:LONG_NLL_CONTINUATION='512'; $env:LONG_NLL_OUT=$out
    $l=& "$Build\tests\research_exl3_long_nll.exe" 2>&1 | Select-String "LONG_NLL|KV_RANGE" | % Line
    "$W $id $(if($L0){'L0'}else{'default'}) $l"
  }
}

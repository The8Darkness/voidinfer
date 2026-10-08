param([int[]]$Windows=@(4096,16384,32768),[string]$Build='D:\AI\voidinfer-sol6-campaign-20260923\builds\B2',[switch]$L0,[string]$Only='')
# Paired NLL for our engine: scores tokens [W, W+512) given tokens [512, W) = NInfer window 2 of --context W --stride 512.
# -L0 runs the engine's long-context mode (L0 OSCAR INT2 history + exact sink/ring window).
$q='D:\AI\engine-compare-20261008\quality'
$receipt=(Get-ChildItem D:\AI\voidinfer-sol6-campaign-20260923\runs -Directory | ? { $_.Name -like '*FastDeviceTransaction-B2' -and (Test-Path (Join-Path $_.FullName 'fast-device-environment.csv')) -and -not (Test-Path (Join-Path $_.FullName 'verify-quality.csv')) } | Sort Name | Select -Last 1).FullName
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
foreach($W in $Windows){
  foreach($ids in Get-ChildItem "$q\w$W\*.tail.ids" | ? { -not $Only -or $_.Name -like "$Only*" }){
    foreach($item in @(Get-ChildItem Env:NINFER* -ErrorAction SilentlyContinue)) { Remove-Item -LiteralPath ("Env:"+$item.Name) }
    foreach($row in (Import-Csv "$receipt\fast-device-environment.csv")) { if($row.Name -notmatch 'L0_OSCAR'){ [Environment]::SetEnvironmentVariable($row.Name,$row.Value,'Process') } }
    if($L0){ $env:NINFER_EXL3_L0_OSCAR='1'; $env:NINFER_EXL3_L0_OSCAR_ROT='D:\AI\kvtier\rot_center384k' }
    $id=$ids.Name -replace '\.tail\.ids$',''
    $out=if($L0){"$q\w$W\$id.ours_l0.tok"}else{"$q\w$W\$id.ours.tok"}
    $env:LONG_NLL_IDS=$ids.FullName; $env:LONG_NLL_CONTEXT=[string]($W-512); $env:LONG_NLL_CONTINUATION='512'; $env:LONG_NLL_OUT=$out
    $l=& "$Build\tests\research_exl3_long_nll.exe" 2>&1 | Select-String "LONG_NLL" | % Line
    "$W $id $(if($L0){'L0'}else{'FP16'}) $l"
  }
}

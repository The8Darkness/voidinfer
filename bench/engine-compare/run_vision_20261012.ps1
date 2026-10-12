# Vision probe 2026-10-12: the two NInfer example images, 2 greedy requests each (256 tokens).
#  1) NInfer ninfer-serve --vision on the NVFP4 artifact (DFlash2, FP8 KV)
#  2) VoidInfer ninfer-serve, EXL3 coherent-device route (default engine) -> media expected to be rejected
#  3) VoidInfer ninfer-serve on the same .ninfer NVFP4 artifact with --vision (route inherited from upstream)
$R='D:\AI\engine-compare-20261008\results\2026-10-12\vision'; New-Item -ItemType Directory -Force $R | Out-Null
$NV='D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer'
function Run-Server($exe,$a,$label,$port){
  $log="$R\$label.server.log"
  $p=Start-Process -FilePath $exe -ArgumentList $a -PassThru -NoNewWindow -WorkingDirectory (Split-Path $exe) -RedirectStandardError $log -RedirectStandardOutput "$log.out"
  try{
    $ok=$false
    for($i=0;$i -lt 900;$i++){ Start-Sleep 1; if($p.HasExited){break}; try{ Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
    if(-not $ok){ "$label server failed to start (exit $($p.ExitCode))"; Get-Content $log -Tail 8; return }
    python D:\AI\engine-compare-20261008\vision_client.py --port $port --label $label --out "$R\$label.json"
  } finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force; Start-Sleep 3 } }
}
Run-Server 'D:\AI\ninfer-5090-windows\build\apps\ninfer-serve.exe' @($NV,'--port','18095','--max-context','32768','--kv-dtype','fp8','--vision','--spec','dflash2','--draft-tokens','7','--lm-head-draft','--greedy','--no-thinking') 'ninfer.nv' 18095
# VoidInfer default (EXL3) route
Set-Location D:\AI\voidinfer-sol6-campaign-20260923
$receipt=(Get-ChildItem runs -Directory | ? { $_.Name -like '*FastDeviceTransaction-B2' -and (Test-Path (Join-Path $_.FullName 'fast-device-environment.csv')) -and -not (Test-Path (Join-Path $_.FullName 'verify-quality.csv')) } | Sort Name | Select -Last 1).FullName
Get-ChildItem Env: | ? { $_.Name -like 'NINFER*' } | % { [Environment]::SetEnvironmentVariable($_.Name,$null,'Process') }
foreach($row in (Import-Csv "$receipt\fast-device-environment.csv")) { if($row.Name -notmatch 'L0_OSCAR|NINFER_TEST|E5A4'){ [Environment]::SetEnvironmentVariable($row.Name,$row.Value,'Process') } }
$env:NINFER_EXL3_L0_OSCAR='1'; $env:NINFER_EXL3_L0_OSCAR_ROT='D:\AI\kvtier\rot_center384k'
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
$ours='D:\AI\voidinfer-sol6-campaign-20260923\builds\B2\apps\ninfer-serve.exe'
Run-Server $ours @('--exl3-target',$env:NINFER_EXL3_TARGET_PATH,'--exl3-draft',$env:NINFER_EXL3_DFLASH2_PATH,'--port','18096','--max-context','32768','--exl3-round','coherent-device','--max-concurrency','1','--greedy','--no-thinking','--model-id','ours','--vision') 'ours.exl3' 18096
Run-Server $ours @('--exl3-target',$env:NINFER_EXL3_TARGET_PATH,'--exl3-draft',$env:NINFER_EXL3_DFLASH2_PATH,'--port','18098','--max-context','32768','--exl3-round','coherent-device','--max-concurrency','1','--greedy','--no-thinking','--model-id','ours') 'ours.exl3-novision' 18098
Get-ChildItem Env: | ? { $_.Name -like 'NINFER*' } | % { [Environment]::SetEnvironmentVariable($_.Name,$null,'Process') }
Run-Server $ours @($NV,'--port','18097','--max-context','32768','--kv-dtype','fp8','--vision','--spec','dflash2','--draft-tokens','7','--lm-head-draft','--greedy','--no-thinking') 'ours.ninfer-nv' 18097
'VISION_DONE'

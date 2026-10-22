# VoidInfer binary on NInfer's NVFP4 .ninfer artifact (route inherited from upstream): DFlash2 on .ninfer is
# rejected by this fork ("--spec dflash2 requires pinned EXL3 K7/full head"), so run it without speculation.
$R='D:\AI\engine-compare-20261008\results\2026-10-12\vision'
$NV='D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer'
$exe='D:\AI\voidinfer-sol6-campaign-20260923\builds\B2\apps\ninfer-serve.exe'
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
Get-ChildItem Env: | ? { $_.Name -like 'NINFER*' } | % { [Environment]::SetEnvironmentVariable($_.Name,$null,'Process') }
$log="$R\ours.ninfer-nv-nospec.server.log"
$p=Start-Process -FilePath $exe -ArgumentList @($NV,'--port','18099','--max-context','32768','--kv-dtype','fp8','--vision','--no-spec','--greedy','--no-thinking') -PassThru -NoNewWindow -WorkingDirectory (Split-Path $exe) -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try{
  $ok=$false; for($i=0;$i -lt 600;$i++){ Start-Sleep 1; if($p.HasExited){break}; try{ Invoke-RestMethod 'http://127.0.0.1:18099/v1/models' -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start"; Get-Content $log -Tail 4; return }
  python D:\AI\engine-compare-20261008\vision_client.py --port 18099 --label ours.ninfer-nv-nospec --out "$R\ours.ninfer-nv-nospec.json"
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force } }

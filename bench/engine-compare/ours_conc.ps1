param([int]$Conc=2,[int]$MaxContext=8192,[int]$Tokens=1024,[string]$Prompt='code_4k',[string]$Env='',[string]$Label='')
# Our side of the concurrency comparison: ninfer-serve (VoidInfer EXL3, coherent-device L0 route)
# with --max-concurrency $Conc, driven by conc_client.py exactly like ninfer_conc.py drives NInfer.
Set-Location D:\AI\voidinfer-sol6-campaign-20260923
$receipt=(Get-ChildItem runs -Directory | ? { $_.Name -like '*FastDeviceTransaction-B2' -and (Test-Path (Join-Path $_.FullName 'fast-device-environment.csv')) -and -not (Test-Path (Join-Path $_.FullName 'verify-quality.csv')) } | Sort Name | Select -Last 1).FullName
Get-ChildItem Env: | ? { $_.Name -like 'NINFER*' } | % { [Environment]::SetEnvironmentVariable($_.Name,$null,'Process') }
foreach($row in (Import-Csv "$receipt\fast-device-environment.csv")) { if($row.Name -notmatch 'L0_OSCAR|NINFER_TEST|E5A4'){ [Environment]::SetEnvironmentVariable($row.Name,$row.Value,'Process') } }
$env:NINFER_EXL3_L0_OSCAR='1'; $env:NINFER_EXL3_L0_OSCAR_ROT='D:\AI\kvtier\rot_center384k'
foreach($kv in ($Env -split ';' | ? {$_})){ $k,$v=$kv -split '=',2; [Environment]::SetEnvironmentVariable($k,$v,'Process') }
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
$port=18433; $log="D:\AI\kvtier\ours_conc.log"
$a=@('--exl3-target',$env:NINFER_EXL3_TARGET_PATH,'--exl3-draft',$env:NINFER_EXL3_DFLASH2_PATH,'--host','127.0.0.1','--port',[string]$port,
 '--max-context',[string]$MaxContext,'--exl3-round','coherent-device','--max-concurrency',[string]$Conc,'--max-pending-requests','8',
 '--no-thinking','--greedy','--temperature','0','--default-max-tokens',[string]$Tokens,'--max-shared-prefixes','16','--host-kv-mib','8000','--model-id','conc')
$p=Start-Process -FilePath .\builds\B2\apps\ninfer-serve.exe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
  $ok=$false
  for($i=0;$i -lt 600;$i++){ Start-Sleep -Milliseconds 1000; if($p.HasExited){break}; try{ Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start"; Get-Content $log -Tail 5; return }
  python D:\AI\engine-compare-20261008\conc_client.py --port $port --conc $Conc --prompt $Prompt --max-tokens $Tokens --label $Label
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force } }

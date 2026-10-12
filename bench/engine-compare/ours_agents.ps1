param([int]$Conc=4,[int]$Agents=8,[int]$Chars=400000,[int]$Tokens=256,[int]$Turns=2,[int]$MaxContext=262144,[int]$HostKvMib=20000,[string]$Out)
# VoidInfer multi-agent long-context run: ninfer-serve (EXL3 coherent-device, L0 OSCAR, exact VeriCache,
# default admission), --max-concurrency $Conc lanes, driven by agents_client.py (same requests as NInfer).
Set-Location D:\AI\voidinfer-sol6-campaign-20260923
$receipt=(Get-ChildItem runs -Directory | ? { $_.Name -like '*FastDeviceTransaction-B2' -and (Test-Path (Join-Path $_.FullName 'fast-device-environment.csv')) -and -not (Test-Path (Join-Path $_.FullName 'verify-quality.csv')) } | Sort Name | Select -Last 1).FullName
Get-ChildItem Env: | ? { $_.Name -like 'NINFER*' } | % { [Environment]::SetEnvironmentVariable($_.Name,$null,'Process') }
foreach($row in (Import-Csv "$receipt\fast-device-environment.csv")) { if($row.Name -notmatch 'L0_OSCAR|NINFER_TEST|E5A4'){ [Environment]::SetEnvironmentVariable($row.Name,$row.Value,'Process') } }
$env:NINFER_EXL3_L0_OSCAR='1'; $env:NINFER_EXL3_L0_OSCAR_ROT='D:\AI\kvtier\rot_center384k'
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
$port=18436; $log="$Out.server.log"
$a=@('--exl3-target',$env:NINFER_EXL3_TARGET_PATH,'--exl3-draft',$env:NINFER_EXL3_DFLASH2_PATH,'--host','127.0.0.1','--port',[string]$port,
 '--max-context',[string]$MaxContext,'--exl3-round','coherent-device','--max-concurrency',[string]$Conc,'--max-pending-requests','16',
 '--no-thinking','--greedy','--temperature','0','--default-max-tokens',[string]$Tokens,'--max-shared-prefixes','16',
 '--host-kv-mib',[string]$HostKvMib,'--pending-timeout-ms','3600000','--model-id','conc')
$p=Start-Process -FilePath .\builds\B2\apps\ninfer-serve.exe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
  $ok=$false; for($i=0;$i -lt 900;$i++){ Start-Sleep 1; if($p.HasExited){break}; try{ Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start"; Get-Content $log -Tail 5; return }
  python D:\AI\engine-compare-20261008\agents_client.py --port $port --agents $Agents --chars $Chars --tokens $Tokens --turns $Turns --out $Out
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force } }

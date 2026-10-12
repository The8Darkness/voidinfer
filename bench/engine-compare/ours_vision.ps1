param([int]$Conc=1,[int]$MaxContext=32768,[string]$Env='',[string]$Out='D:\AI\kvtier\vision_try.json',[switch]$NoVision,[switch]$Hold,[switch]$Suite,[int]$Concurrent=0,[string]$Cases='scene,chart,compare,long_prompt,video,text_after')
# Vision bring-up: our ninfer-serve (coherent-device, L0 OSCAR, exact VeriCache) with the BF16
# Vision tower; a short text prompt, then the two NInfer example images (vision_client.py).
Set-Location D:\AI\voidinfer-sol6-campaign-20260923
$receipt=(Get-ChildItem runs -Directory | ? { $_.Name -like '*FastDeviceTransaction-B2' -and (Test-Path (Join-Path $_.FullName 'fast-device-environment.csv')) -and -not (Test-Path (Join-Path $_.FullName 'verify-quality.csv')) } | Sort Name | Select -Last 1).FullName
Get-ChildItem Env: | ? { $_.Name -like 'NINFER*' } | % { [Environment]::SetEnvironmentVariable($_.Name,$null,'Process') }
foreach($row in (Import-Csv "$receipt\fast-device-environment.csv")) { if($row.Name -notmatch 'L0_OSCAR|NINFER_TEST|E5A4'){ [Environment]::SetEnvironmentVariable($row.Name,$row.Value,'Process') } }
$env:NINFER_EXL3_L0_OSCAR='1'; $env:NINFER_EXL3_L0_OSCAR_ROT='D:\AI\kvtier\rot_center384k'; $env:NINFER_EXL3_VISION_STATS='1'; $env:NINFER_EXL3_VERICACHE_STATS='1'
foreach($kv in ($Env -split ';' | ? {$_})){ $k,$v=$kv -split '=',2; [Environment]::SetEnvironmentVariable($k,$v,'Process') }
$env:PATH='D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin;'+$env:PATH
$port=18440; $log="D:\AI\kvtier\vision_try.log"
$a=@('--exl3-target',$env:NINFER_EXL3_TARGET_PATH,'--exl3-draft',$env:NINFER_EXL3_DFLASH2_PATH,'--host','127.0.0.1','--port',[string]$port,
 '--max-context',[string]$MaxContext,'--exl3-round','coherent-device','--max-concurrency',[string]$Conc,
 '--no-thinking','--greedy','--temperature','0','--model-id','ours')
if(-not $NoVision){ $a+=@('--exl3-vision','D:\AI\models\Qwen3.8-27B-bf16') }
$p=Start-Process -FilePath .\builds\B2\apps\ninfer-serve.exe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
  $base="http://127.0.0.1:$port"; $ok=$false
  for($i=0;$i -lt 900;$i++){ Start-Sleep 1; if($p.HasExited){break}; try{ Invoke-RestMethod "$base/v1/models" -TimeoutSec 2 | Out-Null; $ok=$true; break }catch{} }
  if(-not $ok){ "server failed to start (exit $($p.ExitCode))"; Get-Content $log -Tail 8; return }
  "startup: GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  foreach($q in @('hi','What is 17*23? Answer with the number only.')){
    $body=@{model='ours';max_tokens=64;temperature=0;messages=@(@{role='user';content=$q})} | ConvertTo-Json -Depth 5
    $sw=[Diagnostics.Stopwatch]::StartNew()
    try{ $r=Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
      "text '$q': $($sw.ElapsedMilliseconds) ms pt=$($r.usage.prompt_tokens) ct=$($r.usage.completion_tokens) -> $($r.choices[0].message.content)" }
    catch{ "text '$q' FAILED: $($_.ErrorDetails.Message)" }
  }
  if($Suite){ python D:\AI\engine-compare-20261008\vision_suite.py --port $port --out $Out --cases $Cases --concurrent $Concurrent }
  elseif(-not $NoVision){ python D:\AI\engine-compare-20261008\vision_client.py --port $port --label ours.exl3-vision --out $Out }
  if($Hold){ Read-Host 'hold' }
} finally { if(-not $p.HasExited){ Stop-Process -Id $p.Id -Force; Start-Sleep 2 } ; Get-Content $log | Select-String 'VISION|vision|error|Error|exception' | Select -Last 20 }

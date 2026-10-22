# Resume of the 2026-10-12 queue after the terminal-round VeriCache fix (98f10860): scenarios x2 and base
# (256 outputs, as NInfer) on the fixed binary, then concurrency, multi-agent, vision.
$E='D:\AI\engine-compare-20261008'; $R="$E\results\2026-10-12"; $L="$R\all.log"
function Mark($m){ "$(Get-Date -Format s) $m" | Add-Content $L }
$B="$E\ours_bench.ps1"; $L0Env='NINFER_EXL3_L0_OSCAR=1;NINFER_EXL3_L0_OSCAR_ROT=D:\AI\kvtier\rot_center384k'
$scen=(Get-ChildItem $E\prompts\scenario_*.json | % { $_.BaseName + ':2400' })
New-Item -ItemType Directory -Force "$R\agents","$R\conc" | Out-Null
Mark 'resume: ours scenarios'
foreach($rep in 1,2){ & $B -L0 -Out "$R\ours" -Tag ".rep$rep" -Prompts $scen 2>&1 | Add-Content $L }
Mark 'resume: ours base 256'
& $B -Mode base -Outputs 256 -Out "$R\ours" -Tag '.rep1' -Env $L0Env -Prompts @('code_4k:4096','code_32k:32768','code_128k:131072') 2>&1 | Add-Content $L
Mark 'start concurrency'
& "$E\run_conc_20261012.ps1" 2>&1 | Add-Content $L
Mark 'start ours agents'
& "$E\ours_agents.ps1" -Out "$R\agents\ours.json" 2>&1 | Add-Content $L
Mark 'start ninfer agents'
python "$E\ninfer_agents.py" --artifact D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer --out "$R\agents\ninfer.json" 2>&1 | Add-Content $L
Mark 'start vision'
& "$E\run_vision_20261012.ps1" 2>&1 | Add-Content $L
Mark 'ALL_DONE'

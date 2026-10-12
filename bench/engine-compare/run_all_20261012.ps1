# Remaining GPU queue of the 2026-10-12 comparison, strictly sequential (one GPU job at a time).
$E='D:\AI\engine-compare-20261008'; $R="$E\results\2026-10-12"; $L="$R\all.log"
function Mark($m){ "$(Get-Date -Format s) $m" | Add-Content $L }
New-Item -ItemType Directory -Force "$R\agents" | Out-Null
Mark 'start ninfer 254k small-KV'
& 'C:\Program Files\Git\usr\bin\bash.exe' -lc "cd /d/AI/engine-compare-20261008 && bash run_ninfer_254k_20261012.sh" 2>&1 | Add-Content $L
Mark 'start ours C1'
& "$E\run_ours_20261012.ps1" 2>&1 | Add-Content $L
Mark 'start concurrency'
& "$E\run_conc_20261012.ps1" 2>&1 | Add-Content $L
Mark 'start ours agents'
& "$E\ours_agents.ps1" -Out "$R\agents\ours.json" 2>&1 | Add-Content $L
Mark 'start ninfer agents'
python "$E\ninfer_agents.py" --artifact D:/AI/models/ninfer/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer --out "$R\agents\ninfer.json" 2>&1 | Add-Content $L
Mark 'start vision'
& "$E\run_vision_20261012.ps1" 2>&1 | Add-Content $L
Mark 'ALL_DONE'

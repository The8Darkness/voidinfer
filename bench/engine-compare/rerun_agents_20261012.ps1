$E='D:\AI\engine-compare-20261008'; $A="$E\results\2026-10-12\agents"; $L="$E\results\2026-10-12\all.log"
$GW='D:/AI/models/ninfer/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer'
"$(Get-Date -Format s) start ninfer agents gw.fp8.c4 131072" | Add-Content $L
python "$E\ninfer_agents.py" --artifact $GW --kv fp8 --conc 4 --max-context 131072 --out "$A\ninfer.gw.fp8.c4.ctx131072.json" 2>&1 | Add-Content $L
"$(Get-Date -Format s) start vision inherited" | Add-Content $L
& "$E\run_vision_inherited_20261012.ps1" 2>&1 | Add-Content $L
"$(Get-Date -Format s) RERUN_DONE" | Add-Content $L

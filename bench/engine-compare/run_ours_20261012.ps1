# VoidInfer side of the 2026-10-12 comparison: public Engine via the device harness (warm request of the
# same length, then the measured request), default engine configuration (L0 OSCAR INT4 KV tier with exact
# VeriCache verification), greedy, 1,024 output tokens; base = target-only route with L0, 256 tokens.
$ErrorActionPreference='Continue'
$R='D:\AI\engine-compare-20261008\results\2026-10-12\ours'
$B='D:\AI\engine-compare-20261008\ours_bench.ps1'
$L0Env='NINFER_EXL3_L0_OSCAR=1;NINFER_EXL3_L0_OSCAR_ROT=D:\AI\kvtier\rot_center384k'
$scen=(Get-ChildItem D:\AI\engine-compare-20261008\prompts\scenario_*.json | % { $_.BaseName + ':2400' })
foreach($rep in 1,2){
  & $B -L0 -Out $R -Tag ".rep$rep" -Prompts @('code_4k:4096','code_16k:16384','code_32k:32768','code_64k:65536') | Select -Last 4
}
& $B -L0 -Out $R -Tag '.rep1' -Prompts @('code_128k:131072','code_254k:260000') | Select -Last 2
foreach($rep in 1,2){ & $B -L0 -Out $R -Tag ".rep$rep" -Prompts $scen | Select -Last 12 }
& $B -Mode base -Out $R -Tag '.rep1' -Env $L0Env -Prompts @('code_4k:4096','code_32k:32768','code_128k:131072') | Select -Last 3
'OURS_C1_DONE'

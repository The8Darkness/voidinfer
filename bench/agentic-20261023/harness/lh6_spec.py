"""Long-horizon v6: lh4 (roomly with revisions and deskly interference) with one unrelated side task between every
two stages (bench.side_task: ~100K tokens of legacy source with a planted constant to look up). Each stage boundary
costs one extra compaction, so a run has about 23 compactions instead of 11. Scoring: hidden/lh6 = copy of hidden/lh4;
side-task answers are reported separately."""
from lh4_spec import *  # noqa: F401,F403

SIDE = True

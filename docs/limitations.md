# Limitations

- One model and one GPU class: Qwen3.8-27B as EXL3 6 bpw with its DFlash2 draft, native Windows,
  CUDA `sm_120a` (RTX 5090). Portability beyond that is not qualified.
- Model weights are external and large; reproduction needs the same model directories (target,
  draft, OSCAR rotations, optional BF16 Vision tower), not just a source commit.
- Generation is greedy-first. Positive-temperature sampling runs but is not broadly qualified;
  thinking budgets, stop strings and penalties are rejected by the Engine.
- Media prompts are served without prefix reuse.
- Raw speed trails NInfer (2026-10-12): prefill 1.3-1.8x, base decode 1.2x, batched concurrency
  1.3-1.4x, single-request DFlash2 at 16K-64K 1.1-1.3x, agent resume 2.2x, image requests 1.4-1.6x.
- The resident (pre-L0) OSCAR runtime is still in the source, refused by the Engine and entangled
  with the exact host-state restore path.
- Profiler captures perturb execution; attribution captures are not wall-time shares.
- Passing exactness gates is not broad numerical equivalence or model-quality proof; the KL and
  paired-NLL panels are the quality evidence.

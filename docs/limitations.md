# Limitations

- The current project targets native Windows and CUDA `sm_120a`; portability beyond that is not
  qualified.
- Model weights are external and large. Public reproduction requires matching model manifests and
  revisions, not just a source commit.
- R608 is a research-harness measurement, not HTTP throughput, pure-kernel decode, startup latency,
  or client-visible TTFT.
- EXL3 support is a narrow greedy-text slice. Media, positive-temperature sampling, broad C2+, and
  long-context behavior require separate evidence.
- The held-out quality panel, Mia parity, Engine/client TTFT parity, and the 70/200/3000 performance
  goals are incomplete.
- WMMA32 prefill and native MTP remain default-off and unqualified. R49 and segmented-prefix R612
  remain rejected/default-off for their documented reasons.
- Historical OSCAR/NVFP4/VeriCache results use different model and cache contracts; they are not
  current EXL3 numbers.
- Profiler captures perturb execution. R604 locates a verifier bottleneck but its CUDA API shares
  are not normal wall-time proportions.
- Passing frozen exactness gates is not broad numerical equivalence or model-quality proof.

The project is best presented as a rigorous systems research and implementation portfolio, not as a
drop-in production inference server.

# Resource Scheduling 与 Context Cache 实施记录

本文只记录整体架构的实施步骤和进度；具体设计以
[正式架构文档](resource-scheduling-and-context-cache.md)为准，每一步的执行细节放在独立计划中。

| Step | 内容 | 状态 | 执行计划 |
|---|---|---|---|
| 1 | StateImage Device/Host 物理容器 | 实施完成；focused 5/5、real 3/3 | [StateImage 物理容器实施计划](state-image-container-implementation-plan.md) |
| 2 | Paged KV Device/Host 物理容器 | 实施完成；focused 2/2；27B prefix、35B DFlash 通过，35B MTP 主段完成 | [Paged KV 物理容器实施计划](paged-kv-physical-container-implementation-plan.md) |
| 3 | Engine/resource ownership and context-cache cutover | staged local integration of canonical `dev` through `59febed2`; native Windows Release build and 94-test CTest pass, with the Qwen3.8 NVFP4 Host State/KV, multimodal reuse, cancellation/eviction, and concurrent MTP gate passing | qualify against refreshed canonical `9dbc0740` and retain `9a6813a3` as rollback until independent review; do not cherry-pick isolated resource commits |
| 4 | Context-cost calibration and baseline observability | local Qwen3.8 C1 core matrix, 252,928/262,144 FP8-KV capacity attempts, separate text/Vision serving smokes, and structured baseline report are complete; transfer preset rejected by held-out fit gate | improve the transfer model/fixture without publishing a preset; add concurrent C2/C4/C8 and quality/VLM gates |

The implementation record tracks only accepted/staged architectural milestones. Rejected calibration and
model-specific smoke assumptions remain in `EXPERIMENTS.md`, `KNOWN_ISSUES.md`, and the baseline report.

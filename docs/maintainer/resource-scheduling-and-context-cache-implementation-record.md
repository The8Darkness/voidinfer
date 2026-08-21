# Resource Scheduling 与 Context Cache 实施记录

本文只记录整体架构的实施步骤和进度；具体设计以
[正式架构文档](resource-scheduling-and-context-cache.md)为准，每一步的执行细节放在独立计划中。

| Step | 内容 | 状态 | 执行计划 |
|---|---|---|---|
| 1 | StateImage Device/Host 物理容器 | 实施完成；focused 5/5、real 3/3 | [StateImage 物理容器实施计划](state-image-container-implementation-plan.md) |
| 2 | Paged KV Device/Host 物理容器 | 实施完成；focused 2/2；27B prefix、35B DFlash 通过，35B MTP 主段完成 | [Paged KV 物理容器实施计划](paged-kv-physical-container-implementation-plan.md) |

后续步骤在范围确定后追加。每一步完成时，仅在这里补充完成状态、提交和验证结论。

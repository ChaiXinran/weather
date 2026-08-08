# 历史版本记录

本目录保存已经完成或已终止的研究版本记录，用来回答“当时使用了什么协议、改了什么、跑出了什么结果、为什么继续修改”。这里只保存小型文字和机器可读索引，不复制 checkpoint、大型逐样本表或图片；原始产物继续留在 `work_dirs/`，并由各版本文档链接。

## 命名规则

`V<序号>_<阶段>_<简短名称>.md`

- `completed`：训练和预定评估均完成，可以引用结果。
- `partial`：只完成部分训练或评估，不得作为正式结论。
- `superseded`：历史上有效，但协议已经更新，跨版本比较必须注明差异。

## 版本索引

| 版本 | 日期 | 状态 | 内容 |
|---|---|---|---|
| [data-cache-20260808](data_cache_20260808.md) | 2026-08-08 | completed | PWV/RAIN 无损 uint8 mmap 缓存与读取接入 |
| [V0.1](V0.1_radar_simvp_30epoch_seed0.md) | 2026-07-30 | completed | 首轮正式 Radar-only Tiny SimVP，30 epoch，seed 0 |

未来每次冻结一次模型、数据协议、损失函数或评估协议，都应新增版本文件，不覆盖旧记录；同时更新 `index.yml`、`.research/experiment_matrix.yml` 和 `.research/run_log.md`。

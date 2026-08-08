# 2026-08-08 — PWV/RAIN 无损帧缓存

## 状态

`completed`。参照 Radar 的 uint8 NPY + manifest + mmap 方案，为 2025 PWV 和 RAIN 建立一次性无损缓存，并将 RAIN 缓存接入现有数据加载器。

## 数据结果

- PWV：29,516 个唯一时间帧，形状 `[29516,66,70]`，`frames.npy` 136,364,048 字节。
- RAIN：29,511 个唯一时间帧，形状 `[29511,66,70]`，`frames.npy` 136,340,948 字节。
- 两类原图均为 70×66、L 模式、uint8；2025 年内未发现重复时间戳。
- 缓存保存原始反灰度像素，不固化归一化、物理解码或 RAIN 时空对齐。

## 代码改动

- `tools/cache_bth_png.py`：PWV/RAIN 通用构建器。
- `tools/verify_bth_png_cache.py`：像素一致性和实际归一化读取微基准。
- `openstl/datasets/png_cache.py`：变量与形状校验、时间戳索引、每 worker 延迟 mmap。
- `openstl/datasets/dataloader_radar.py`：新增 `rain_cache_path`，RAIN truth 支持缓存和 PNG 回退。
- `openstl/datasets/dataloader.py`：透传 `rain_cache_path`。
- `configs/bth_radar/SimVP_gSTA_smoke_rain_truth.py`：启用 `RAIN_CACHE_UINT8`。
- `tests/test_datasets/test_radar.py`：增加 RAIN PNG/cache 等值回归测试。

## 验证

- PWV 和 RAIN 各抽查 23 帧，全部逐像素完全一致。
- 600 帧 float32 反灰度归一化微基准：PWV 6.49×，RAIN 8.60×。
- WSL OpenSTL 环境运行 `tests/test_datasets/test_radar.py`：4 passed。
- pytest 仅报告旧版 scikit-image 的 `np.bool8` 弃用警告，以及沙箱下无法写 `.pytest_cache`；不影响测试结论。

## 路径与边界

- `/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S/PWV_CACHE_UINT8`
- `/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S/RAIN_CACHE_UINT8`
- 原始 PNG 全部保留。
- 性能数字是读取微基准，不宣称端到端训练按同倍数加速。


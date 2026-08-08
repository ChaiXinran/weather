# BTH PWV/RAIN 无损缓存

## 目的

PWV 和 RAIN 与 Radar 一样由大量 70×66 的单通道 PNG 组成。训练或评估时逐张目录查找、打开和解压会重复消耗时间，因此将每个变量的唯一时间帧保存为连续的 uint8 NPY，并用 JSON 保存时间戳索引。缓存不预展开样本窗口，不改变原 PNG，也不提前执行归一化、RAIN 对齐或其他研究协议。

## 已生成的 2025 缓存

```text
DATA_2025_S/
├── PWV_CACHE_UINT8/
│   ├── frames.npy       # [29516, 66, 70], uint8
│   └── manifest.json
└── RAIN_CACHE_UINT8/
    ├── frames.npy       # [29511, 66, 70], uint8
    └── manifest.json
```

- PWV NPY：136,364,048 字节（含 NPY 头）。
- RAIN NPY：136,340,948 字节（含 NPY 头）。
- PWV 解码语义：`PWV_mm=(255-pixel)*80/255`。
- RAIN 解码语义：`rain_mm_h=(255-pixel)*35/255`。
- 模型常用的 0–1 输入仍是 `(255-pixel)/255`。

## 构建

在 WSL OpenSTL 环境、仓库根目录执行：

```bash
export PYTHONPATH=.
python tools/cache_bth_png.py \
  --variable pwv \
  --data-root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S

python tools/cache_bth_png.py \
  --variable rain \
  --data-root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S
```

已有完整缓存时脚本会拒绝覆盖；只有明确传入 `--overwrite` 才会重建。写入先使用 `frames.npy.tmp` 和 `manifest.json.tmp`，全部成功后再原子发布正式文件。

## 读取

RAIN truth 配置增加：

```python
evaluation_truth = 'rain_png'
rain_cache_path = 'RAIN_CACHE_UINT8'
```

未提供 `rain_cache_path` 时，数据集自动回退到原 PNG。RAIN 的时间滞后和行列平移仍在 `rain_targets()` 中执行，不固化到缓存。

未来 PWV loader 可直接复用：

```python
from openstl.datasets.png_cache import BTHPNGCache

cache = BTHPNGCache(cache_path, 'pwv', expected_size=(70, 66))
pixels = cache.read(timestamp)  # float32 raw inverse-grayscale pixels
pwv_normalized = (255.0 - pixels) / 255.0
```

`BTHPNGCache` 校验 format、variable、uint8 dtype、帧数和空间形状，并由每个 DataLoader worker 延迟打开只读 mmap。

## 2026-08-08 验证

命令：

```bash
python tools/verify_bth_png_cache.py --cache /path/to/PWV_CACHE_UINT8
python tools/verify_bth_png_cache.py --cache /path/to/RAIN_CACHE_UINT8
```

固定 seed 42，对每类首/中/末帧加 20 个随机帧进行比较：PWV 23/23、RAIN 23/23 均逐像素完全一致。

600 帧读取并转换成 float32 0–1 数组的只读微基准：

| 变量 | PNG | mmap cache | 加速 |
|---|---:|---:|---:|
| PWV | 4.822870 s | 0.742755 s | 6.49× |
| RAIN | 1.339572 s | 0.155704 s | 8.60× |

这是 I/O 与归一化微基准，不等于端到端训练加速；训练还受 GPU、模型计算、DataLoader worker 和磁盘缓存状态影响。


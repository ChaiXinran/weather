# 项目上下文总索引

> 这是新对话的首要入口。最后更新：2026-07-31。研究协议详见 `.research/detail.md`；机器可读数据与实验记录见同目录 YAML 文件。
> 历史版本从 [`.research/history/README.md`](history/README.md) 进入；已冻结的首轮 30 epoch 基线记录为
> [`V0.1_radar_simvp_30epoch_seed0.md`](history/V0.1_radar_simvp_30epoch_seed0.md)。

## 1. 研究目标与路线

项目研究京津冀区域多源、物理可解释的短临降水预报。固定任务是用过去 10 帧 Radar/PWV/DEM（60 分钟）预测未来 20 帧 Radar（120 分钟），时间间隔 6 分钟，空间网格 66×70、约 0.1°。当前先建立 Radar-only 强基线，再依次研究简单融合、运动—源汇分解、PWV 对源汇的动态调制、DEM 静态地形适配、跨年份/区域泛化和高级对象演化。

核心科学问题不是只降低格点误差，而是改善强降水的生成、增强、维持和位移，同时避免长时效弱降水面积膨胀。PWV 需通过真实、静态或打乱等对照证明动态信息有效；DEM 需通过消融证明地形贡献；最终模型应轻量、稳定、可解释、可泛化。

## 2. 数据、输入输出与缓存

- 输入张量 `[B,10,1,66,70]`，目标 `[B,20,1,66,70]`。Radar PNG 为反灰度，`dBZ=(255-pixel)*50/255`，训练归一化为 `(255-pixel)/255`。
- 2025：65 个事件、17,359 个样本；train 56/11,245，val 4/932，test 5/5,182；Radar 32,457 帧，Rain 29,511 帧。
- 2023：59 个事件、8,204 个样本；train 42/4,921，val 8/1,868，test 9/1,415；当前只作外年份诊断。2023 Rain 稀疏，且 2023-07-29 存在重复时间戳。
- 事件判据：20 dBZ、湿区比例 0.01、最大干间隔 3 小时、前后扩展 30 分钟。正式样本 stride=1。
- 2025 原始数据：`/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S`。
- Radar 缓存：上述目录下 `RADAR_CACHE_UINT8/frames.npy + manifest.json`，形状 `[32457,66,70]`、uint8、约 144 MiB。它是无损帧缓存，使用 mmap、每 worker 延迟打开；归一化仍在加载时执行。100 个 10→20 样本的 I/O 从 11.727 s 降至 0.069 s（168.99×，不代表端到端训练加速）。
- 正式评估将未来 Radar 用冻结的 Marshall–Palmer `Z=200R^1.6` 转成雨强。Rain PNG 只用于 Z-R 选择和诊断；其量化且封顶 35 mm/h，并非独立站点真值。Rain 对齐固定为 +42 分钟、行 0、列 +1。

## 3. 环境与本机调用

- Windows 仓库：`D:\_Search\AIforScience\Rewritten\origin\OpenSTL`
- WSL 发行版：`Ubuntu-D`；WSL 仓库：`/mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL`
- Conda 环境：`OpenSTL`；Python：`/home/ranye/miniconda3/envs/OpenSTL/bin/python`
- GPU：RTX 4060 Laptop，8 GB。
- 已确认版本：Python 3.10.8，PyTorch 2.1.1+cu118，torchvision 0.16.1+cu118，CUDA 11.8，cuDNN 8.7，OpenCV 4.8.1，Lightning 2.2.1，NumPy 1.26.4，Pillow 12.2.0，Matplotlib 3.10.9，pandas 2.2.3，scikit-image 0.19.3，timm 0.9.16。
- 已安装 PyCINRAD 1.9.3、Cartopy；此前安装使 xarray 更新到 2025.6.1。`cnmaps` 与 `pytest` 未确认安装。
- 从仓库运行必须先 `export PYTHONPATH=.`，否则可能导入环境中旧版 OpenSTL，出现不认识 `bth_radar` 的问题。
- 典型入口：`wsl -d Ubuntu-D`，进入仓库，`conda activate OpenSTL`，`export PYTHONPATH=.`，然后运行 `python tools/train.py ...` 或 `python tools/test.py ...`。更完整命令保存在根目录 `detail.md`。
- `--fp16` 虽能被解析，但当前 Trainer 未实际接线 precision；当前训练按 fp32 记录。Tensor Core 的 matmul precision 提示尚未处理。

## 4. 代码架构

- `openstl/datasets/dataloader_radar.py`：严格连续的 10→20 序列、manifest split、PNG/缓存读取、Rain 时间/空间对齐。
- `openstl/datasets/radar_protocol.py`：事件识别、manifest 与 Z-R 辅助逻辑。
- `openstl/datasets/dataloader.py`、`__init__.py`、`dataset_constant.py`：`bth_radar` 注册和装载。
- `openstl/core/precipitation_metrics.py`：五层流式降水评估。
- `openstl/methods/base_method.py`：当前 MSE、验证降水指标、测试评估器与可选 Rain truth。
- `openstl/api/exp.py`：实验生命周期、双 checkpoint、显式 `ckpt_path` 测试加载。
- `openstl/utils/callbacks.py`：checkpoint 别名。
- `configs/bth_radar/`：Radar 配置；主配置为 `SimVP_gSTA.py`。
- `tools/prepare_data/build_bth_protocol.py`、`calibrate_bth_zr_v2.py`、`cache_bth_radar.py`：数据协议、Z-R 与缓存。
- `tools/visualizations/generate_radar_sequence.py`：Radar 时序可视化。
- `tools/train.py`、`tools/test.py`：训练/测试入口；`tests/` 含 Radar 与降水指标测试。

## 5. 模型、损失与训练指标

当前模型是 Tiny SimVP-gSTA：`hid_S=32, hid_T=128, N_S=2, N_T=4, kernel=3`，约 330 万参数，权重约 13.171 MiB，checkpoint 约 39.67 MiB。原模型自然输出 10 帧；现用两次递归 10→10 得到 20 帧，第二段以第一段预测作为输入，容易累积误差与平滑。

当前损失严格是归一化 Radar 张量上的 `nn.MSELoss()`。它让背景和弱降水占主导，是目前强回波消失、面积/能量不足的主要嫌疑。强降水加权或多阈值损失仍属计划，尚未实现或冻结。

默认优化：Adam、lr=1e-3、weight decay=0、OneCycle、train batch=16、workers=4、确定性 seed。正式运行请求 val batch=16，但实际参数记录为 8，需排查配置合并。

训练记录至少包含 `train_loss`、`val_loss`。R1 新增验证期 CSI/POD/FAR/Bias/AreaRatio/IntensityRatio，并定义：

`val_csi_score = CSI16(0-1h) + CSI32(0-1h) + CSI16(1-2h) + 2×CSI32(1-2h)`。

checkpoint 分别保存 `best_val_loss.ckpt`、`best_val_csi.ckpt`、`last.ckpt`。

## 6. 冻结评估体系

五层体系：数据质量、像素评分、空间结构、对象演化、统计可信度。

- 雨强阈值：0.1、2.5、8、16、32 mm/h；16/32 是强降水主指标。
- 时效：每个 lead（+6 至 +120 分钟）；快照 +30/+60/+90/+120；分段 0–1 h、1–2 h、全时段。
- 连续指标：MAE、RMSE、Mean Error、Intensity Ratio；同时报告全有效格点和观测湿区。
- 分类指标：TP/FP/FN/TN、POD、FAR、CSI、HSS、Frequency Bias。分母为零时写 NaN/null，macro 时跳过。
- 空间指标：严格 1×1 为主；FSS 1/3/5、质心误差、雨区面积比、能量比、peak/p95/p99、PSD/高频保持为诊断。
- 对象指标：16/32 阈值连通域、IoU≥0.1 匹配、对象 POD/FAR/IoU、质心、面积、峰值。轨迹寿命与 split/merge 尚未完整实现。
- 统计：micro 汇总；event macro 的 mean/median/q25/q75；按事件配对 bootstrap 2,000 次、seed 42、95% CI；正式结论计划 seed 0/1/2 的 mean±std。
- persistence 必须与模型使用完全相同的 mask、阈值和统计口径。
- 输出包括 summary、逐 lead/window/event/object 表、混淆计数、bootstrap、lead 曲线、PSD、典型成功/失败案例及预测/真值/persistence 并排图。

内部晋级参考：1–2 h 的 ΔCSI32>0 且 bootstrap 下界>0；首小时 ΔCSI 不低于 -0.005；面积比约 0.8–1.2 且 FAR 不恶化；对象、质心、能量/峰值改善；三个 seed 方向一致。

## 7. 可视化现状

已有随机样本时序图，可观察回波移动、强度变化、生成和消散；使用 PyCINRAD 反射率色标，并增加差值层、20/35/45 dBZ 面积与行列方向说明。`color.pdf` 主要介绍 Matplotlib LightSource 与 cnmaps/Cartopy 地形渲染，不是 Radar 色标规范；Radar 色标已采用 PyCINRAD。正式评估还生成逐时效曲线、PSD、成功/失败案例以及 truth/model/persistence 并排图。

## 8. 已完成正式结果

`bth_simvp_gsta_formal_seed0`：30 epoch、seed 0、stride 1。训练约 1 h 55 min，完整测试约 40 min。train loss 从 0.018725 降至 0.001702；val loss 早期最佳 0.0103946，末期为 0.0139349，说明过拟合。

最后 checkpoint 相对 persistence：MAE 0.450829 vs 0.634024，RMSE 2.445422 vs 3.148734，连续误差更好；但 intensity ratio 0.549 vs 1.000。模型 CSI 在 0.1/2.5/8/16/32 阈值分别为 0.6002/0.3427/0.1442/0.0575/0.0168；persistence 为 0.5730/0.3180/0.1772/0.1131/0.0693。模型只在弱阈值占优，强降水显著落后。

按 val loss 选择的 best checkpoint：MAE 0.435604、RMSE 2.485620、intensity ratio 0.36393；CSI 为 0.6498/0.2460/0.0856/0.0319/0.0102。它比 last 更严重地压低强降水。因此“val-loss best”不是“降水技巧 best”。

结论：Tiny SimVP+MSE 能改善平均连续误差，却通过平滑和低估牺牲强雨面积、能量和 CSI；当前基线没有通过强降水晋级标准。

## 9. 历史版本、改动和未完成状态

- 历史版本使用 `.research/history/` 单独维护，版本文档只保存协议、变更、结果与原始产物链接，不复制大型 checkpoint 和指标表。
- 最早线性趋势 smoke（211 样本）比 persistence 差，验证了数据与评估链路。
- `smoke_10ep_s10` 和 `smoke_10ep_s10_rain_truth` 完成了端到端与 Rain 对齐诊断，但协议/尺度与当前正式协议不同，数值不可直接横比。
- 旧 positive-only 本地 Z-R：`a=42.1157,b=0.9921`；zero-aware 候选：`a=285.3878,b=0.7929,z0=4`；均因验证集连续或分类表现不如 Marshall–Palmer 被拒绝。完整证据在 `local_zr*.json` 和 `zr_protocol_decision.json`。
- 当前工作树相对上游单一导入提交 `a65210d` 含尚未提交的 R1 修改：验证降水评分、双 checkpoint、显式 checkpoint 测试加载。不要把这些本地改动误认为已进入 Git 历史。
- `bth_simvp_gsta_r1_5ep_seed0` 计划 5 epoch，但只观察到完成 epoch 0–2；best CSI 在 epoch 0（0.331482），best val loss 在 epoch 1（0.010082）。没有完整测试报告，必须标记为“未完成”，不能据此下结论。

## 10. 下一步建议顺序

1. 先确认 R1 是否重跑并完成，验证降水评分 checkpoint 是否确实优于 val-loss checkpoint。
2. 修复/确认 val batch-size 与 fp16 配置问题，并冻结一条可复现训练命令。
3. 设计强雨友好的损失，先用短程消融，再跑完整 seed 0/1/2；必须同时检查 FAR、面积比、能量比和长时效。
4. Radar-only 稳定后再接 PWV，做真实/静态/打乱 PWV 消融；之后接 DEM。
5. 增加独立站点验证或明确 Radar-derived rain 的结论边界。

# OpenSTL 本机环境记录

更新时间：2026-07-31

## 运行环境

- Windows 仓库：`D:\_Search\AIforScience\Rewritten\origin\OpenSTL`
- WSL 发行版：`Ubuntu-D`
- WSL 仓库：`/mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL`
- Conda 环境名：`OpenSTL`
- 环境 Python：`/home/ranye/miniconda3/envs/OpenSTL/bin/python`
- 数据目录：`/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S`
- Radar 缓存：数据目录下的 `RADAR_CACHE_UINT8`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）

Windows 默认的 `python` 不是本项目训练环境，不应直接用来训练或验证。
该环境当前没有安装 `pytest`，可以先用 `compileall` 和训练 smoke run
检查；需要跑 pytest 时，应在 `OpenSTL` 环境中安装 pytest。

配置文件使用：

```python
radar_cache_path = 'RADAR_CACHE_UINT8'
```

它会相对 `data_root` 解析，实际对应 Windows 路径
`D:\_Search\AIforScience\Rewritten\capsule-3935105\data\DATA_2025_S\RADAR_CACHE_UINT8`
（WSL 路径为
`/mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S/RADAR_CACHE_UINT8`）。
这是可直接训练的 uint8 NPY 缓存，不需要逐张读取 Radar PNG。

## 从 PowerShell 调用

查看 WSL 发行版：

```powershell
wsl --list --verbose
```

检查项目 Python、PyTorch 和 CUDA：

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  env PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python -c `
  "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

执行 Python 语法检查：

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  env PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python -m compileall -q `
  openstl/methods/base_method.py openstl/api/exp.py openstl/utils/callbacks.py
```

## R1：5-epoch 启动命令

R1 保持 Tiny-SimVP、MSE、递归 10→20、数据划分、batch size、学习率和
seed 不变，只增加验证期强降水指标以及 MSE-best/CSI-best checkpoint。

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  env PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python tools/train.py `
  --dataname bth_radar `
  --method SimVP `
  --config_file configs/bth_radar/SimVP_gSTA.py `
  --data_root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S `
  --ex_name bth_simvp_gsta_r1_5ep_seed0 `
  --epoch 5 `
  --batch_size 16 `
  --val_batch_size 8 `
  --seed 0 `
  --deterministic `
  --no_display_method_info
```

必须设置 `PYTHONPATH=.`。否则 `tools/train.py` 会优先导入 Conda 环境中安装的
旧版 `openstl` 包，而不是当前工作区代码；旧版解析器也不认识 `bth_radar`。

实验输出目录：

```text
work_dirs/bth_simvp_gsta_r1_5ep_seed0
```

R1 checkpoint：

```text
checkpoints/best_val_loss.ckpt
checkpoints/best_val_csi.ckpt
checkpoints/last.ckpt
```

注意：`tools/train.py` 训练后还会自动运行一次完整 test。若只需要训练和
验证曲线，可在训练完成后终止 test；正式比较时应显式指定要评估的 checkpoint。

## R2：5-epoch 启动命令

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  env PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python tools/train.py `
  --dataname bth_radar `
  --method SimVP `
  --config_file configs/bth_radar/SimVP_gSTA_r2.py `
  --data_root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S `
  --ex_name bth_simvp_gsta_r2_5ep_seed0 `
  --epoch 5 `
  --batch_size 16 `
  --val_batch_size 8 `
  --seed 0 `
  --deterministic `
  --no_display_method_info
```

R2a/R2b/R2d 消融队列：

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  bash tools/run_bth_r2_ablations.sh
```

脚本会自动设置 `PYTHONPATH=.`，按 R2a、R2b、R2d 顺序各训练 5 epoch，
并使用 `--skip_test_after_train` 跳过训练后的隐式 Last 测试。

## R3：直接 10→20

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  env PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python tools/train.py `
  --dataname bth_radar `
  --method SimVP `
  --config_file configs/bth_radar/SimVP_gSTA_r3.py `
  --data_root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S `
  --ex_name bth_simvp_gsta_r3_direct_5ep_seed0 `
  --epoch 5 `
  --batch_size 16 `
  --val_batch_size 8 `
  --seed 0 `
  --deterministic `
  --no_display_method_info `
  --skip_test_after_train
```

显式验证 checkpoint 并生成逐 lead 指标：

```powershell
wsl -d Ubuntu-D --cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL `
  env PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python tools/validate.py `
  --dataname bth_radar --method SimVP `
  --config_file configs/bth_radar/SimVP_gSTA_r2d.py `
  --data_root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S `
  --ex_name bth_simvp_gsta_r2d_best_lead_validation `
  --batch_size 16 --val_batch_size 8 --epoch 5 --seed 0 --deterministic `
  --no_display_method_info `
  --ckpt_path work_dirs/bth_simvp_gsta_r2d_5ep_seed0/checkpoints/best_val_csi.ckpt
```

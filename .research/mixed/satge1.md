# bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0 Validation Report

Date: 2026-08-07

## 1. Executive Summary

- Work directory: `/root/weather/work_dirs/bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0`
- Checkpoint: `/root/weather/work_dirs/bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0/checkpoints/best_val_csi.ckpt`
- Method: `directphysicshybrid`
- Configuration: `configs/bth_radar/DirectPhysicsHybrid_r2d_no_deep_convlstm.py`
- Weighted validation CSI score: **0.937194**
- Validation MSE: **0.01097923**
- Evaluation split: Validation only; the Test split was not used.

The weighted score is not a single CSI percentage. It is:

```text
CSI16(0-1h) + CSI32(0-1h) + CSI16(1-2h) + 2 * CSI32(1-2h)
```

## 2. Model and Protocol

| Item | Value |
|---|---:|
| Total parameters | 4,809,641 |
| Trainable parameters | 1,064,617 |
| Frozen parameters | 3,745,024 |
| Batch size | 4 |
| Validation batch size | 4 |
| Seed | 0 |
| Epochs configured | 10 |
| Loss type | precipitation_r2 |
| Input/output frames | 10 / 20 |

## 3. Training Curve

| Epoch | LR | Train loss | Validation loss | Weighted CSI | Wall (s) | Peak GPU (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | n/a | n/a | n/a | 0.935945 | n/a | n/a |
| 1 | 0.0000760 | 0.519064 | 0.011540 | 0.936199 | 242.3 | 1780.5 |
| 2 | 0.0001000 | 0.177826 | 0.010979 | 0.937194 | 242.9 | 1780.6 |
| 3 | 0.0000950 | 0.177650 | 0.011179 | n/a | 250.4 | 1780.6 |
| 4 | 0.0000812 | 0.176890 | 0.011328 | n/a | 240.0 | 1780.6 |
| 5 | 0.0000611 | 0.175299 | 0.010988 | n/a | 244.3 | 1780.6 |
| 6 | 0.0000389 | 0.173030 | 0.011148 | n/a | 242.2 | 1780.6 |
| 7 | 0.0000188 | 0.171606 | 0.011107 | n/a | 220.3 | 1780.6 |
| 8 | 0.0000049 | 0.170740 | 0.011195 | n/a | 244.1 | 1780.6 |
| 9 | 0.0000000 | 0.170267 | 0.011188 | n/a | 242.0 | 1780.6 |

## 4. Period Metrics

| Metric | First hour | Second hour |
|---|---:|---:|
| CSI at 16 mm/h | 0.382616 | 0.139123 |
| CSI at 32 mm/h | 0.272644 | 0.071406 |
| POD at 16 mm/h | 0.555716 | 0.298206 |
| POD at 32 mm/h | 0.521497 | 0.240486 |
| FAR at 16 mm/h | 0.448765 | 0.793154 |
| FAR at 32 mm/h | 0.636394 | 0.907802 |
| Bias at 16 mm/h | 1.008130 | 1.441686 |
| Bias at 32 mm/h | 1.434236 | 2.608369 |
| Area ratio at 16 mm/h | 1.008130 | 1.441686 |
| Area ratio at 32 mm/h | 1.434236 | 2.608369 |
| Intensity ratio | 0.996991 | 1.197864 |

## 5. Lead-Time Metrics

| Lead | CSI16 | CSI32 | POD16 | POD32 | FAR16 | FAR32 | Bias16 | Bias32 | Intensity ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 min | 0.746001 | 0.666311 | n/a | n/a | 0.129958 | 0.206215 | 0.964957 | 1.015127 | 0.989305 |
| 12 min | 0.616490 | 0.512126 | n/a | n/a | 0.216850 | 0.342374 | 0.949228 | 1.061869 | 0.978991 |
| 18 min | 0.516471 | 0.411678 | n/a | n/a | 0.296663 | 0.452205 | 0.938834 | 1.138391 | 0.973995 |
| 24 min | 0.442420 | 0.338588 | n/a | n/a | 0.368438 | 0.542915 | 0.944217 | 1.239064 | 0.974011 |
| 30 min | 0.384648 | 0.281003 | n/a | n/a | 0.433254 | 0.618185 | 0.961390 | 1.350307 | 0.977884 |
| 36 min | 0.335633 | 0.239215 | n/a | n/a | 0.494950 | 0.674953 | 0.990276 | 1.462301 | 0.985883 |
| 42 min | 0.296405 | 0.200833 | n/a | n/a | 0.547541 | 0.726885 | 1.021505 | 1.579727 | 0.997410 |
| 48 min | 0.264158 | 0.172373 | n/a | n/a | 0.593581 | 0.766404 | 1.058240 | 1.698441 | 1.011641 |
| 54 min | 0.236601 | 0.149372 | n/a | n/a | 0.635365 | 0.798981 | 1.104027 | 1.828910 | 1.030879 |
| 60 min | 0.213774 | 0.130061 | n/a | n/a | 0.670481 | 0.826032 | 1.148164 | 1.954804 | 1.049731 |
| 66 min | 0.194672 | 0.114092 | n/a | n/a | 0.700372 | 0.848417 | 1.192225 | 2.082509 | 1.072750 |
| 72 min | 0.178017 | 0.099479 | n/a | n/a | 0.727409 | 0.868321 | 1.244013 | 2.196051 | 1.094293 |
| 78 min | 0.162333 | 0.088678 | n/a | n/a | 0.752591 | 0.883510 | 1.296192 | 2.324947 | 1.121368 |
| 84 min | 0.150351 | 0.078450 | n/a | n/a | 0.772375 | 0.897488 | 1.348481 | 2.443603 | 1.148461 |
| 90 min | 0.139958 | 0.071016 | n/a | n/a | 0.790109 | 0.907833 | 1.409324 | 2.564087 | 1.178624 |
| 96 min | 0.131204 | 0.065413 | n/a | n/a | 0.804749 | 0.915665 | 1.463262 | 2.676599 | 1.208798 |
| 102 min | 0.123219 | 0.059906 | n/a | n/a | 0.818328 | 0.923204 | 1.524260 | 2.787639 | 1.239547 |
| 108 min | 0.117253 | 0.056292 | n/a | n/a | 0.828882 | 0.928288 | 1.586039 | 2.893325 | 1.272089 |
| 114 min | 0.111948 | 0.053439 | n/a | n/a | 0.838188 | 0.932340 | 1.646804 | 2.995989 | 1.304959 |
| 120 min | 0.107056 | 0.049803 | n/a | n/a | 0.846631 | 0.937255 | 1.706548 | 3.099908 | 1.337089 |

## 6. All Validation Metrics

| Metric | Value |
|---|---:|
| `val_area_ratio_16_0_1h` | 1.0081296708 |
| `val_area_ratio_16_1_2h` | 1.4416857492 |
| `val_area_ratio_32_0_1h` | 1.4342355464 |
| `val_area_ratio_32_1_2h` | 2.6083685444 |
| `val_bias_16_0_1h` | 1.0081296708 |
| `val_bias_16_1_2h` | 1.4416857492 |
| `val_bias_16_lead_006m` | 0.9649565744 |
| `val_bias_16_lead_012m` | 0.9492283210 |
| `val_bias_16_lead_018m` | 0.9388342892 |
| `val_bias_16_lead_024m` | 0.9442170684 |
| `val_bias_16_lead_030m` | 0.9613897663 |
| `val_bias_16_lead_036m` | 0.9902760589 |
| `val_bias_16_lead_042m` | 1.0215053763 |
| `val_bias_16_lead_048m` | 1.0582401776 |
| `val_bias_16_lead_054m` | 1.1040274380 |
| `val_bias_16_lead_060m` | 1.1481640270 |
| `val_bias_16_lead_066m` | 1.1922252754 |
| `val_bias_16_lead_072m` | 1.2440125044 |
| `val_bias_16_lead_078m` | 1.2961917484 |
| `val_bias_16_lead_084m` | 1.3484806456 |
| `val_bias_16_lead_090m` | 1.4093236800 |
| `val_bias_16_lead_096m` | 1.4632620105 |
| `val_bias_16_lead_102m` | 1.5242600863 |
| `val_bias_16_lead_108m` | 1.5860390020 |
| `val_bias_16_lead_114m` | 1.6468036818 |
| `val_bias_16_lead_120m` | 1.7065483245 |
| `val_bias_32_0_1h` | 1.4342355464 |
| `val_bias_32_1_2h` | 2.6083685444 |
| `val_bias_32_lead_006m` | 1.0151268485 |
| `val_bias_32_lead_012m` | 1.0618689436 |
| `val_bias_32_lead_018m` | 1.1383907424 |
| `val_bias_32_lead_024m` | 1.2390642959 |
| `val_bias_32_lead_030m` | 1.3503072871 |
| `val_bias_32_lead_036m` | 1.4623010521 |
| `val_bias_32_lead_042m` | 1.5797267281 |
| `val_bias_32_lead_048m` | 1.6984414886 |
| `val_bias_32_lead_054m` | 1.8289103337 |
| `val_bias_32_lead_060m` | 1.9548041513 |
| `val_bias_32_lead_066m` | 2.0825093533 |
| `val_bias_32_lead_072m` | 2.1960509639 |
| `val_bias_32_lead_078m` | 2.3249467519 |
| `val_bias_32_lead_084m` | 2.4436030291 |
| `val_bias_32_lead_090m` | 2.5640872621 |
| `val_bias_32_lead_096m` | 2.6765991259 |
| `val_bias_32_lead_102m` | 2.7876386688 |
| `val_bias_32_lead_108m` | 2.8933254266 |
| `val_bias_32_lead_114m` | 2.9959892169 |
| `val_bias_32_lead_120m` | 3.0999081606 |
| `val_csi_16_0_1h` | 0.3826155280 |
| `val_csi_16_1_2h` | 0.1391225979 |
| `val_csi_16_lead_006m` | 0.7460011217 |
| `val_csi_16_lead_012m` | 0.6164900080 |
| `val_csi_16_lead_018m` | 0.5164707045 |
| `val_csi_16_lead_024m` | 0.4424201529 |
| `val_csi_16_lead_030m` | 0.3846483170 |
| `val_csi_16_lead_036m` | 0.3356328073 |
| `val_csi_16_lead_042m` | 0.2964048109 |
| `val_csi_16_lead_048m` | 0.2641579616 |
| `val_csi_16_lead_054m` | 0.2366010553 |
| `val_csi_16_lead_060m` | 0.2137737435 |
| `val_csi_16_lead_066m` | 0.1946722719 |
| `val_csi_16_lead_072m` | 0.1780174696 |
| `val_csi_16_lead_078m` | 0.1623331887 |
| `val_csi_16_lead_084m` | 0.1503514211 |
| `val_csi_16_lead_090m` | 0.1399584636 |
| `val_csi_16_lead_096m` | 0.1312035782 |
| `val_csi_16_lead_102m` | 0.1232190773 |
| `val_csi_16_lead_108m` | 0.1172534060 |
| `val_csi_16_lead_114m` | 0.1119480464 |
| `val_csi_16_lead_120m` | 0.1070555596 |
| `val_csi_32_0_1h` | 0.2726439472 |
| `val_csi_32_1_2h` | 0.0714058168 |
| `val_csi_32_lead_006m` | 0.6663114202 |
| `val_csi_32_lead_012m` | 0.5121260312 |
| `val_csi_32_lead_018m` | 0.4116779843 |
| `val_csi_32_lead_024m` | 0.3385877693 |
| `val_csi_32_lead_030m` | 0.2810026871 |
| `val_csi_32_lead_036m` | 0.2392152870 |
| `val_csi_32_lead_042m` | 0.2008334117 |
| `val_csi_32_lead_048m` | 0.1723725301 |
| `val_csi_32_lead_054m` | 0.1493723963 |
| `val_csi_32_lead_060m` | 0.1300606899 |
| `val_csi_32_lead_066m` | 0.1140918530 |
| `val_csi_32_lead_072m` | 0.0994790830 |
| `val_csi_32_lead_078m` | 0.0886782173 |
| `val_csi_32_lead_084m` | 0.0784497285 |
| `val_csi_32_lead_090m` | 0.0710158211 |
| `val_csi_32_lead_096m` | 0.0654130289 |
| `val_csi_32_lead_102m` | 0.0599061310 |
| `val_csi_32_lead_108m` | 0.0562924562 |
| `val_csi_32_lead_114m` | 0.0534389517 |
| `val_csi_32_lead_120m` | 0.0498034736 |
| `val_csi_score` | 0.9371937066 |
| `val_far_16_0_1h` | 0.4487653393 |
| `val_far_16_1_2h` | 0.7931543584 |
| `val_far_16_lead_006m` | 0.1299581371 |
| `val_far_16_lead_012m` | 0.2168498363 |
| `val_far_16_lead_018m` | 0.2966630786 |
| `val_far_16_lead_024m` | 0.3684379515 |
| `val_far_16_lead_030m` | 0.4332536074 |
| `val_far_16_lead_036m` | 0.4949500102 |
| `val_far_16_lead_042m` | 0.5475413887 |
| `val_far_16_lead_048m` | 0.5935808475 |
| `val_far_16_lead_054m` | 0.6353647951 |
| `val_far_16_lead_060m` | 0.6704812529 |
| `val_far_16_lead_066m` | 0.7003721560 |
| `val_far_16_lead_072m` | 0.7274090587 |
| `val_far_16_lead_078m` | 0.7525909506 |
| `val_far_16_lead_084m` | 0.7723753600 |
| `val_far_16_lead_090m` | 0.7901086509 |
| `val_far_16_lead_096m` | 0.8047489309 |
| `val_far_16_lead_102m` | 0.8183277880 |
| `val_far_16_lead_108m` | 0.8288822790 |
| `val_far_16_lead_114m` | 0.8381875536 |
| `val_far_16_lead_120m` | 0.8466312057 |
| `val_far_32_0_1h` | 0.6363939821 |
| `val_far_32_1_2h` | 0.9078019947 |
| `val_far_32_lead_006m` | 0.2062145005 |
| `val_far_32_lead_012m` | 0.3423739630 |
| `val_far_32_lead_018m` | 0.4522054453 |
| `val_far_32_lead_024m` | 0.5429148251 |
| `val_far_32_lead_030m` | 0.6181854556 |
| `val_far_32_lead_036m` | 0.6749527279 |
| `val_far_32_lead_042m` | 0.7268853856 |
| `val_far_32_lead_048m` | 0.7664043033 |
| `val_far_32_lead_054m` | 0.7989813492 |
| `val_far_32_lead_060m` | 0.8260318548 |
| `val_far_32_lead_066m` | 0.8484167977 |
| `val_far_32_lead_072m` | 0.8683211324 |
| `val_far_32_lead_078m` | 0.8835098769 |
| `val_far_32_lead_084m` | 0.8974881748 |
| `val_far_32_lead_090m` | 0.9078331480 |
| `val_far_32_lead_096m` | 0.9156647370 |
| `val_far_32_lead_102m` | 0.9232044722 |
| `val_far_32_lead_108m` | 0.9282883950 |
| `val_far_32_lead_114m` | 0.9323399026 |
| `val_far_32_lead_120m` | 0.9372553169 |
| `val_intensity_ratio_0_1h` | 0.9969908540 |
| `val_intensity_ratio_1_2h` | 1.1978640868 |
| `val_intensity_ratio_lead_006m` | 0.9893046530 |
| `val_intensity_ratio_lead_012m` | 0.9789907283 |
| `val_intensity_ratio_lead_018m` | 0.9739947838 |
| `val_intensity_ratio_lead_024m` | 0.9740106516 |
| `val_intensity_ratio_lead_030m` | 0.9778835237 |
| `val_intensity_ratio_lead_036m` | 0.9858831102 |
| `val_intensity_ratio_lead_042m` | 0.9974099878 |
| `val_intensity_ratio_lead_048m` | 1.0116410349 |
| `val_intensity_ratio_lead_054m` | 1.0308794123 |
| `val_intensity_ratio_lead_060m` | 1.0497310634 |
| `val_intensity_ratio_lead_066m` | 1.0727501203 |
| `val_intensity_ratio_lead_072m` | 1.0942927399 |
| `val_intensity_ratio_lead_078m` | 1.1213675157 |
| `val_intensity_ratio_lead_084m` | 1.1484611103 |
| `val_intensity_ratio_lead_090m` | 1.1786235694 |
| `val_intensity_ratio_lead_096m` | 1.2087977186 |
| `val_intensity_ratio_lead_102m` | 1.2395467062 |
| `val_intensity_ratio_lead_108m` | 1.2720886140 |
| `val_intensity_ratio_lead_114m` | 1.3049593619 |
| `val_intensity_ratio_lead_120m` | 1.3370886040 |
| `val_loss_epoch` | 0.0109792277 |
| `val_pod_16_0_1h` | 0.5557160170 |
| `val_pod_16_1_2h` | 0.2982064137 |
| `val_pod_32_0_1h` | 0.5214966757 |
| `val_pod_32_1_2h` | 0.2404863769 |

## 7. Checkpoint Inventory

| Checkpoint | Size (MiB) |
|---|---:|
| `best_val_csi.ckpt` | 26.64 |
| `best_val_loss.ckpt` | 26.64 |
| `last.ckpt` | 26.64 |
| `val-csi-epoch=00-val_csi_score=0.935945.ckpt` | 26.64 |
| `val-csi-epoch=01-val_csi_score=0.936199.ckpt` | 26.64 |
| `val-csi-epoch=02-val_csi_score=0.937194.ckpt` | 26.64 |
| `val-loss-epoch=02-val_loss=0.010979.ckpt` | 26.64 |

## 8. Configuration Snapshot

| Key | Value |
|---|---|
| `aft_seq_length` | `20` |
| `batch_size` | `4` |
| `bootstrap_repetitions` | `2000` |
| `bootstrap_seed` | `42` |
| `case_count` | `3` |
| `case_threshold` | `32.0` |
| `ckpt_path` | `/root/weather/work_dirs/bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0/checkpoints/best_val_csi.ckpt` |
| `clip_grad` | `None` |
| `clip_mode` | `norm` |
| `config_file` | `configs/bth_radar/DirectPhysicsHybrid_r2d_no_deep_convlstm.py` |
| `convert_dbz_to_rain` | `True` |
| `data_root` | `/root/weather/data` |
| `dataname` | `bth_radar` |
| `decay_epoch` | `100` |
| `decay_rate` | `0.1` |
| `deterministic` | `True` |
| `device` | `cuda` |
| `dist` | `False` |
| `drop` | `0.0` |
| `drop_last` | `False` |
| `drop_path` | `0.0` |
| `epoch` | `10` |
| `evolution_encoder_checkpoint` | `None` |
| `evolution_encoder_lr` | `None` |
| `evolution_field_space` | `None` |
| `evolution_freeze_encoder_epochs` | `None` |
| `evolution_freeze_motion_epochs` | `None` |
| `evolution_gate_active_threshold` | `None` |
| `evolution_gate_initial` | `None` |
| `evolution_gate_lr` | `None` |
| `evolution_gate_supervision_only` | `False` |
| `evolution_gate_supervision_weight` | `None` |
| `evolution_head_lr` | `None` |
| `evolution_motion_checkpoint` | `None` |
| `evolution_stop_gradient` | `False` |
| `evolution_use_flow_gate` | `False` |
| `ex_name` | `bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0_auto_report_val` |
| `filter_bias_and_bn` | `False` |
| `filter_size` | `5` |
| `final_div_factor` | `10000.0` |
| `fp16` | `False` |
| `fps` | `False` |
| `gpus` | `[0]` |
| `grid_spacing_km` | `10.0` |
| `hybrid_alpha_max` | `0.08` |
| `hybrid_alpha_regularization` | `0.01` |
| `hybrid_blend_warmup_epochs` | `0` |
| `hybrid_convlstm_kernel` | `3` |
| `hybrid_convlstm_scales` | `[]` |
| `hybrid_direct_anchor_after_warmup` | `0.02` |
| `hybrid_direct_anchor_weight` | `0.1` |
| `hybrid_direct_checkpoint` | `work_dirs/bth_convlstm_r2d_ft3ep_seed0/checkpoints/best_val_csi.ckpt` |
| `hybrid_flow_regularization` | `0.0001` |
| `hybrid_fpn_channels` | `96` |
| `hybrid_freeze_direct` | `True` |
| `hybrid_gate_lr_scale` | `0.05` |
| `hybrid_gate_regularization` | `0.002` |
| `hybrid_gate_supervision_weight` | `0.1` |
| `hybrid_gate_temperature` | `0.05` |
| `hybrid_head_channels` | `64` |
| `hybrid_lead_channels` | `8` |
| `hybrid_max_residual_displacement` | `2.0` |
| `hybrid_max_source_rain` | `12.0` |
| `hybrid_motion_alpha_max` | `0.5` |
| `hybrid_physics_aux_weight` | `0.1` |
| `hybrid_residual_aux_weight` | `0.1` |
| `hybrid_source_alpha_max` | `0.25` |
| `hybrid_source_regularization` | `1e-05` |
| `hybrid_temporal_kernel` | `3` |
| `hybrid_temporal_mix_scales` | `[0, 1, 2, 3]` |
| `hybrid_unet_blocks` | `[1, 1, 2, 2]` |
| `hybrid_unet_channels` | `[32, 64, 128, 192]` |
| `hybrid_warmup_physics_weight` | `1.0` |
| `in_shape` | `[10, 1, 66, 70]` |
| `init_from_ckpt` | `None` |
| `layer_norm` | `0` |
| `lead_minutes` | `6` |
| `log_step` | `1` |
| `loss_type` | `precipitation_r2` |
| `lr` | `0.0001` |
| `lr_k_decay` | `1.0` |
| `manifest_path` | `.research/bth_2025_events.json` |
| `method` | `directphysicshybrid` |
| `metric_for_bestckpt` | `val_loss` |
| `metrics` | `['mae', 'rmse']` |
| `min_lr` | `1e-06` |
| `model_type` | `gSTA` |
| `momentum` | `0.9` |
| `motion_eval_only` | `False` |
| `neighborhood_windows` | `[1, 3, 5]` |
| `no_display_method_info` | `True` |
| `num_hidden` | `64,64,64,64` |
| `num_workers` | `4` |
| `object_iou_threshold` | `0.1` |
| `opt` | `adam` |
| `opt_betas` | `None` |
| `opt_eps` | `None` |
| `overwrite` | `False` |
| `patch_size` | `2` |
| `pre_seq_length` | `10` |
| `precip_clip_range` | `[0.0, 50.0]` |
| `precip_thresholds` | `[0.1, 2.5, 8.0, 16.0, 32.0]` |
| `precip_value_unit` | `mm/h` |
| `r2_empty_event_penalty` | `0.1` |
| `r2_huber_beta` | `0.05` |
| `r2_intensity_weights` | `[2.0, 3.0]` |
| `r2_second_hour_weight` | `1.2` |
| `r2_segmented_soft_csi_weights` | `[[0.0018, 0.0009], [0.00216, 0.00108]]` |
| `r2_soft_csi_mode` | `sample_period` |
| `r2_soft_csi_temperature` | `0.03` |
| `r2_soft_csi_weights` | `[0.005, 0.001]` |
| `r2_thresholds` | `[16.0, 32.0]` |
| `radar_cache_path` | `RADAR_CACHE_UINT8` |
| `radar_value_scale` | `50.0` |
| `res_dir` | `work_dirs` |
| `reverse_scheduled_sampling` | `0` |
| `sampling_changing_rate` | `2e-05` |
| `sampling_start_value` | `1.0` |
| `sampling_stop_iter` | `50000` |
| `sched` | `onecycle` |
| `scheduled_sampling` | `1` |
| `seed` | `0` |
| `skip_test_after_train` | `True` |
| `stride` | `1` |
| `test` | `False` |
| `torchscript` | `False` |
| `total_length` | `30` |
| `use_augment` | `False` |
| `use_prefetcher` | `False` |
| `val_batch_size` | `4` |
| `val_precip_thresholds` | `[16.0, 32.0]` |
| `warmup_epoch` | `0` |
| `warmup_lr` | `1e-05` |
| `weight_decay` | `0.0` |
| `wet_threshold` | `0.1` |
| `zr_a` | `200.0` |
| `zr_b` | `1.6` |
| `zr_fit_artifact` | `.research/local_zr_v2.json` |
| `zr_selection` | `marshall_palmer_validation_winner` |

# DirectPhysicsHybrid Branch Attribution

Checkpoint: `/root/weather/work_dirs/bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0/checkpoints/best_val_csi.ckpt`

Validation only; the Test split was not used.

## Branch Metrics

| Branch | Period | Threshold | CSI | POD | FAR | Bias |
|---|---|---:|---:|---:|---:|---:|
| direct | 0_1h | 16 | 0.380877 | 0.550659 | 0.447365 | 0.996425 |
| direct | 0_1h | 32 | 0.271653 | 0.519130 | 0.637007 | 1.430136 |
| direct | 1_2h | 16 | 0.138186 | 0.292411 | 0.792392 | 1.408475 |
| direct | 1_2h | 32 | 0.070976 | 0.236923 | 0.907992 | 2.575016 |
| motion_only | 0_1h | 16 | 0.382438 | 0.555065 | 0.448493 | 1.006451 |
| motion_only | 0_1h | 32 | 0.272643 | 0.521207 | 0.636254 | 1.432887 |
| motion_only | 1_2h | 16 | 0.139047 | 0.297909 | 0.793179 | 1.440420 |
| motion_only | 1_2h | 32 | 0.071406 | 0.240400 | 0.907789 | 2.607077 |
| source_only | 0_1h | 16 | 0.380996 | 0.551214 | 0.447674 | 0.997988 |
| source_only | 0_1h | 32 | 0.271719 | 0.519521 | 0.637079 | 1.431498 |
| source_only | 1_2h | 16 | 0.138239 | 0.292658 | 0.792397 | 1.409700 |
| source_only | 1_2h | 32 | 0.070992 | 0.237069 | 0.907986 | 2.576440 |
| fused | 0_1h | 16 | 0.382616 | 0.555716 | 0.448765 | 1.008130 |
| fused | 0_1h | 32 | 0.272644 | 0.521497 | 0.636394 | 1.434236 |
| fused | 1_2h | 16 | 0.139123 | 0.298206 | 0.793154 | 1.441686 |
| fused | 1_2h | 32 | 0.071406 | 0.240486 | 0.907802 | 2.608369 |

## Event Transitions Relative to Direct

| Branch | Period | Threshold | Miss->Hit | Hit->Miss | FA->Correct | Correct->FA |
|---|---|---:|---:|---:|---:|---:|
| motion_only | 0_1h | 16 | 3055 | 1310 | 2798 | 5024 |
| motion_only | 0_1h | 32 | 1013 | 705 | 2438 | 2538 |
| motion_only | 1_2h | 16 | 2561 | 381 | 2165 | 12652 |
| motion_only | 1_2h | 32 | 663 | 138 | 2181 | 6497 |
| source_only | 0_1h | 16 | 221 | 1 | 0 | 399 |
| source_only | 0_1h | 32 | 58 | 0 | 0 | 144 |
| source_only | 1_2h | 16 | 98 | 0 | 2 | 390 |
| source_only | 1_2h | 32 | 22 | 0 | 0 | 193 |
| fused | 0_1h | 16 | 3176 | 1173 | 2571 | 5204 |
| fused | 0_1h | 32 | 1031 | 680 | 2352 | 2609 |
| fused | 1_2h | 16 | 2634 | 336 | 1977 | 12848 |
| fused | 1_2h | 32 | 669 | 131 | 2085 | 6583 |

# Pi0.5 JAX：双臂、三相机、100 点动作

这一目录维护独立的 JAX 训练和模型服务代码。机器人客户端继续使用原来的 `emimic` 环境、相机 SHM、键盘、动作调度和日志。
环境由用户配置；本次实现不创建 `.venv`、不安装依赖、不修改参考项目的环境。`pyproject.toml` 是依赖配方，新环境的安装结果需要由用户运行后确认。

## 数据与时间约定

- 原始转换 Zarr 保持一个采集控制 tick 一行，不改成 100 Hz，也不重编码图像。
- `piper_pico_to_zarr` 可同时导出 Pi0.5 manifest；已经转换的数据也可单独补导出。
- **插值发生在 `Pi05Dataset.__getitem__`**：从当前行取源动作窗口，在 episode 尾部重复最后一行，应用 stride，逐臂变换到训练坐标系，再将位置/夹爪线性插值、姿态 SLERP 到 **100 点**，最后转成 Euler。
- 默认 `source_fps=30, source_horizon=30, action_stride=1`，首末跨度为 `29/30=0.966667 s`。这是与原 Piper loader 一致的名义频率/行索引约定，不声称实际采集时间戳完全等间隔。
- state 来自 SDK 实际反馈；action 来自遥操下发的目标。每臂顺序为 `xyz[m], yaw/pitch/roll[rad, intrinsic ZYX], normalized_gripper`，先左后右，共 14 维。
- 使用每条 episode 的 `T_base_camera`，没有固定单位外参或单臂增量控制假设。当前单位外参数据保持原有数值约定。
- 图像在原转换时已经完成内参处理；JAX 数据加载不再次去畸变。训练和服务共用 `ResizeImages(224,224)` 等比缩放/补边、tokenizer、归一化和输入适配。
- 先对物理 14 维计算新的训练集统计、归一化，再补零到模型的 32 维；输出反归一化后取回 14 维。
- `front_1/left_wrist/right_wrist` 分别映射到 `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`。
- manifest 使用相对路径引用 Zarr，**不能只复制 manifest 而遗漏源 Zarr**。metadata 与数值数据参与身份校验，JPEG 内容不逐字节做哈希。

当前实现完成 Piper 数据训练与部署；100 点表示和逐 episode 坐标处理为后续 ego 混合保留一致的入口。**尚未实现人手数据混合采样、缺失 wrist 图像 mask 和人手动作适配**，不能把任意 ego Zarr 直接交给本版 loader。

## 1. 用户配置 JAX 环境

参考项目为 `/home/yesen/cys/avfrl_real/pi05`。OpenPI 固定为：

```text
c23745b5ad24e98f66967ea795a07b2588ed6c79
```

Python 3.11、JAX 0.6.2、Flax 0.10.2、Orbax 0.11.13；CPU Torch 仅用于数据加载，JAX 使用 GPU。完整版本约束见 `pyproject.toml`。
若用这份配方创建独立环境，由用户执行以下命令（本次开发没有执行）：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
# 目标 pi05/openpi 尚不存在时执行；不修改参考 checkout。
git clone --no-hardlinks /home/yesen/cys/avfrl_real/pi05/openpi pi05/openpi
git -C pi05/openpi checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
cd pi05
uv sync --python 3.11
```

后续命令使用绝对解释器路径，避免把 JAX 依赖装进 `emimic`。也可通过 `--openpi-root` 指向同版本的已有只读 OpenPI checkout。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
PI05_PYTHON=/home/yesen/pico_ego_probe/EgoVerse/pi05/.venv/bin/python
PI05_OPENPI=/home/yesen/pico_ego_probe/EgoVerse/pi05/openpi
PI05_PARAMS=/home/yesen/cys/avfrl_real/pi05/.cache/openpi/pi05_base/params
PI05_TOKENIZER=/home/yesen/cys/avfrl_real/pi05/.cache/openpi/big_vision/paligemma_tokenizer.model
PI05_MANIFEST=/home/yesen/pico_ego_probe/EgoVerse/data/pi05/1107_pick/manifest.json
PI05_STATS=/home/yesen/pico_ego_probe/EgoVerse/data/pi05/1107_pick/stats
```

## 2. 从已有转换数据导出

不需要重新采集或重新转换视频。默认按完整 episode 分出 10% 验证集，不能把同一 episode 的不同窗口混入两边。

```bash
python -m egomimic.scripts.export_piper_pi05 \
  --input /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted/1107_pick \
  --output "$PI05_MANIFEST" \
  --source-fps 30 --source-horizon 30 --action-stride 1 \
  --validation-ratio 0.1 --seed 42
```

也可在下一批原始采集数据转换时，一次完成 Zarr 与 Pi0.5 manifest：

```bash
python -m egomimic.scripts.piper_pico_to_zarr \
  --input /绝对路径/新采集批次 \
  --output-dir /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted \
  --calibration /home/yesen/pico_ego_probe/EgoVerse/calibrations/lrcp_front_480p_identity_extrinsics_001.json \
  --width 640 --height 480 --undistort-alpha 0 \
  --fps 30 --max-frame-age 0.21 \
  --task-name pick_place \
  --task-description 'Pick up the object and place it in the container.' \
  --pi05-manifest /绝对路径/新批次_manifest.json \
  --pi05-source-horizon 30 --pi05-action-stride 1 \
  --pi05-validation-ratio 0.1 --pi05-split-seed 42
```

保留原转换器“不覆盖已有 episode”的行为；已有批次请用独立导出命令。不同标定的 Piper episode 可同时存在，逐条使用其自身外参。此版要求三路图像输出尺寸和夹爪开度定义兼容。

## 3. 归一化与输入检查

正式训练的统计不要传 `--max-windows`。该参数仅用于快速测试，并会在报告里标记为 `smoke_test_statistics`。
统计只读取训练 episode，将 100 个时间点合并计算每个物理维度的分位数，不沿用 HPT 的统计文件。

```bash
"$PI05_PYTHON" pi05/scripts/compute_pi05_stats.py \
  --manifest "$PI05_MANIFEST" --output-dir "$PI05_STATS" \
  --openpi-root "$PI05_OPENPI"

"$PI05_PYTHON" pi05/scripts/train_pi05.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" --exp-name pick_pi05_lora \
  --batch-size 1 --dry-run
```

`--dry-run` 使用 CPU 检查真实图像、任务/state tokenization、`100×14 → 100×32`，不分配完整模型权重。

## 4. 必须先跑的 LoRA 测试

这是实际完整预训练权重、三相机、100 点动作测试，不是缩小的假模型。显存占用和首次编译时间不能当作稳定训练速度。

```bash
PI05_PROBE=/home/yesen/pico_ego_probe/EgoVerse/diagnostics/pi05_lora_my_test

"$PI05_PYTHON" pi05/scripts/check_pi05_lora.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" --run-dir "$PI05_PROBE" --inspect

CUDA_VISIBLE_DEVICES=0 "$PI05_PYTHON" pi05/scripts/check_pi05_lora.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" --run-dir "$PI05_PROBE" --steps 2

CUDA_VISIBLE_DEVICES=0 "$PI05_PYTHON" pi05/scripts/check_pi05_lora.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" --run-dir "$PI05_PROBE" --steps 4 --resume
```

检查语言 LoRA、动作专家 LoRA、全部可训练参数与优化器实际更新；冻结参数每个叶子取固定样本做校验（不是全部冻结权重逐字节哈希）。恢复时逐项比较保存的状态。结果在 `lora_result*.json`，保留的每个 checkpoint 都带完整推理资产。

## 5. 正式训练与恢复

以下是可运行的配置模板，不是已调好的超参数或已完成的任务效果评估：

```bash
CUDA_VISIBLE_DEVICES=0 "$PI05_PYTHON" pi05/scripts/train_pi05.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" \
  --checkpoint-root /home/yesen/pico_ego_probe/EgoVerse/pi05/checkpoints \
  --exp-name pick_pi05_lora --finetune-mode lora \
  --batch-size 1 --num-workers 4 \
  --train-steps 10000 --learning-rate 1e-5 \
  --warmup-steps 100 --decay-steps 10000 --end-learning-rate 1e-6 \
  --save-interval 1000 --keep-period 1000
```
CUDA_VISIBLE_DEVICES=0 "$PI05_PYTHON" pi05/scripts/train_pi05.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" \
  --checkpoint-root /home/yesen/pico_ego_probe/EgoVerse/pi05/checkpoints \
  --exp-name pick_pi05_full --finetune-mode full \
  --batch-size 1 --num-workers 4 \
  --train-steps 20000 --learning-rate 1e-5 \
  --warmup-steps 100 --decay-steps 20000 --end-learning-rate 1e-6 \
  --save-interval 1000 --keep-period 1000



恢复时保持数据、统计、tokenizer、batch 和优化器配置一致，添加 `--resume`，`--train-steps` 为新的**总更新次数**。
恢复参数、优化器、随机状态和步数；上游 DataLoader 重新开始采样，不保证与未中断运行的逐 batch 顺序完全相同。
`--finetune-mode full` 提供全参数模式，需要另行评估显存。当前没有自动在线验证循环；使用下一节的离线验证。

OpenPI checkpoint 目录使用循环编号：完成 10000 次更新通常为 `9999/`。每个已提交 checkpoint 自带 `assets/egoverse/{contract.json,norm_stats.json,tokenizer.model}`，服务端不依赖训练 Zarr。

## 6. 离线恢复与预测检查

```bash
PI05_CHECKPOINT="$PI05_PROBE/checkpoints/egoverse_pi05_lora/lora_probe/3"
CUDA_VISIBLE_DEVICES=0 "$PI05_PYTHON" pi05/scripts/check_pi05_inference.py \
  --checkpoint-path "$PI05_CHECKPOINT" --manifest "$PI05_MANIFEST" \
  --openpi-root "$PI05_OPENPI" --split validation --sample-index 0 \
  --output "$PI05_PROBE/inference_validation.json"
```

不连接机器人。严格加载 LoRA，检查真实训练/验证输入输出 `100×14`，记录 XYZ、SO(3) 旋转、夹爪误差、二阶差分、同输入重复采样差异，以及预热后的推理耗时；同时保存预测和标签 NPZ。
这里只检查一个窗口，`passed` 表示加载和数值链路通过，**不表示模型已学会任务或抖动已解决**。2–4 步测试权重不能用于真机任务。

## 7. 服务与机器人客户端

模型终端使用 JAX 环境，指定正式训练权重：

```bash
CUDA_VISIBLE_DEVICES=0 "$PI05_PYTHON" pi05/scripts/serve_pi05.py \
  --checkpoint-path /绝对路径/正式训练/数字checkpoint目录 \
  --openpi-root "$PI05_OPENPI" \
  --num-inference-steps 10 --host 127.0.0.1 --port 8000
```

可先追加 `--check-only` 在 CPU 验证资产哈希和配置。服务正式启动会先加载模型、预热，再开放连接。

机器人终端继续 `emimic`，下面命令会连接相机和机械臂，但没有 `--execute` 时不下发策略动作：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.infer_piper_policy \
  --policy-backend pi05 --server-url ws://127.0.0.1:8000 --server-timeout 10 \
  --training-data /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted/1107_pick \
  --calibration /home/yesen/pico_ego_probe/EgoVerse/calibrations/lrcp_front_480p_identity_extrinsics_001.json \
  --camera-config /home/yesen/pico_ego_probe/EgoVerse/calibrations/uvc_rig.json \
  --task-description 'Pick up the object and place it in the container.' \
  --control-frequency 30 --execute-steps 30 --no-log
```

正式执行时添加 `--execute`，按需添加 `--home-on-start true`。键盘空格开始/暂停、Q 退出。客户端不需要 PICO 头显或 JAX；只需现有 `websocket-client` 和 `msgpack`。
握手核对 100 点时间配置、标定、图像处理、夹爪和单位；命令行仍显式提供训练用内外参 JSON。RPC 同步请求、超时不自动重发；返回动作仍走原来的坐标逆变换、夹爪范围处理和时间重采样。
100 ms 动作调度检查保持原样。RPC 耗时会增加 chunk 间等待，本版没有 RTC 或跨 chunk 平滑。

## 8. 回归测试

```bash
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_pi05_data.py tests/unit/test_pi05_jax_config.py \
  tests/unit/test_pi05_remote.py tests/unit/test_piper_pico_to_zarr.py \
  tests/unit/test_live_policy.py tests/unit/test_policy_execution.py
```

`tests/integration/test_pi05_jax_lora.py` 可一次运行 CPU 检查、GPU 两步训练、恢复到四步、严格推理。使用 `EGOVERSE_TEST_PI05_{PYTHON,OPENPI_ROOT,MANIFEST,STATS_DIR,TOKENIZER,PARAMS}` 指定路径，并传 `pytest --integration -m gpu`；默认跳过。
`tests/integration/test_pi05_server_client.py` 使用 `EGOVERSE_TEST_PI05_{PYTHON,OPENPI_ROOT,CHECKPOINT,MANIFEST}`，临时启动真实模型服务，从 `emimic` 客户端读取离线样本验证 RPC 和 100→30，无硬件连接。

## 本次验证记录（2026-11-08）

未创建/安装新环境；使用已有 `/home/yesen/cys/avfrl_real/pi05/.venv/bin/python` 和本地预训练权重做验证。
结果目录：`diagnostics/pi05_100point_20261108/`，总表为 `verification.json`。

- 44 条真实 episode 导出，40 条训练、4 条验证；原数据未修改。
- CPU 输入检查通过：三路 224×224 RGB、100×14 物理动作、100×32 模型输入。
- 353 个相关 CPU 回归通过，4 个旧 PyTorch Pi 可选依赖测试跳过；随后新增服务摘要测试通过，JAX 资产组现为 11 项通过。
- RTX 5090 D 上完成完整 Pi0.5 权重加载、LoRA 两步训练、恢复到第四步，语言/动作专家 LoRA 与优化器更新检查通过。
- 保存后严格恢复与验证集单窗口推理通过，返回 100×14。仅是冒烟测试，使用 32 个训练窗口统计；这些权重和误差不代表正式训练质量。
- 真实 JAX server 与 `emimic` client 往返通过：三路处理后图像逐像素一致，state 最大绝对差约 2.22e-16；100 点按原时间跨度恢复成 30 点；错误外参在推理前被拒绝。
- 测试服务已关闭，没有连接相机/CAN 或执行机械臂动作。新环境安装和正式长时间训练仍由用户开展。

服务加载时先导入正常模型依赖，再初始化/检查 JAX GPU，避免本机旧系统 CUDA 路径使服务入口过早加载不匹配的库；没有修改系统库或旧环境文件。

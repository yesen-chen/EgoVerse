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

Piper 单独训练仍沿用上述路径。PICO Ego 与 Piper 混合训练现已提供独立导出入口，支持各自的源动作窗口、Ego 完整未来窗口限制、分域归一化及缺失腕部图像 mask。详见 [PICO Ego 混合训练操作说明](README_pico_ego_mixed.md)。混合模型的机器人部署继续使用 Piper 时间和反归一化统计。

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
100 ms 动作调度检查保持原样。上面的同步模式中，RPC 耗时会增加 chunk 间等待；可选 RTC 模式见下文。

## 8. 回归测试

```bash
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' EGOVERSE_TEST_PI05_PYTHON="$PWD/pi05/.venv/bin/python" python -m pytest -q \
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

## 完整离线误差、抖动与静止保持评估

`evaluate_pi05.py` 从历史 HPT 离线评估读取同一批 episode/frame，使用 Pi0.5 训练清单里的三图和实际反馈 state，经真实模型服务得到动作。检查 state/标签一致性，保留原始 100 点，再用生产坐标转换及重采样恢复 30 Hz。无需相机或机械臂。

先按上面的服务命令启动待测 checkpoint（`--num-inference-steps 10`），然后在仓库根目录、`emimic` 环境运行：

```bash
python pi05/scripts/evaluate_pi05.py \
  --checkpoint-path "$PI05_CHECKPOINT" \
  --manifest "$PI05_MANIFEST" --openpi-root "$PI05_OPENPI" \
  --reference-eval diagnostics/piper_offline_eval_20261107_152801_fc1d54 \
  --server-url ws://127.0.0.1:8000 --num-inference-steps 10 \
  --output-dir diagnostics/pi05_offline_new_run

python pi05/scripts/check_pi05_hold.py \
  --evaluation diagnostics/pi05_offline_new_run \
  --openpi-root "$PI05_OPENPI" --server-url ws://127.0.0.1:8000 \
  --per-side 24 --repeats 5 --worst-repeats 20

python pi05/scripts/report_pi05_eval.py \
  --evaluation diagnostics/pi05_offline_new_run

python -m pytest -q tests/unit/test_pi05_evaluation.py
```

评估完成后在服务终端按 Ctrl+C 关闭临时服务。模型服务和客户端各用已有环境，不需要往 JAX 环境安装机器人/Aria 依赖。

静止窗口要求命令、实际反馈都基本不动，并排除 episode 末尾补帧；重复测试按左右臂分别跨 episode 选取。报告分开列出内部细碎变化、相对当前姿态的偏移，以及固定输入的随机预测分散。离线固定输入不能验证长期闭环保持；训练清单未划分验证集时也不能据此判断泛化能力。

2026-11-08 的 full checkpoint `12000` 评估保存在 `diagnostics/pi05_full_12000_eval_20261108/report_zh.md`。包含 352 个同输入对照、左右各 24 个代表静止输入，以及各臂最差案例的 20 次复查。右臂个别静止输入会预测明显主动位移，具体证据保存在 `hold_audit.json` 和 `hold_repeats/`，不可只看平均抖动指标。

## RTC：在模型内部约束相邻动作段

RTC 是可选的推理方式，不需要重新训练，现有 checkpoint、归一化、标定和 **100 点训练动作**保持不变。实现依据 [Physical Intelligence RTC 论文](https://www.pi.website/download/real_time_chunking.pdf) 的推理期前缀引导：客户端在执行旧段时获取新图像和实际反馈，将旧段剩余目标按新观测时间对齐到 100 点模型网格；服务器在每个 flow step 内通过完整 VJP 约束去噪后的动作。硬前缀覆盖计划接管 tick、一个控制周期的漂移裕量，以及该时刻右侧的模型插值点；这些有效重叠点权重为 1，之后按论文 Eq.5 衰减，旧段范围之外为 0。

当前客户端默认在模型 RTC 之外增加**执行连续性层**：`--rtc-smoothing-ms 100` 对完整新预测做居中 Savitzky–Golay 位姿滤波；当前100点时间网格对应11点、3阶，实际窗口约97.6ms。位置直接滤波，旋转先转为符号一致的单位四元数，滤波后重新归一化。`--rtc-transition-ms 200` 在同一绝对时刻的旧、新计划之间做五次 smoothstep 位置过渡和 SO(3) 短弧过渡；在底层插值可微处衔接入口、出口速度，结束后直接使用新计划。100点时间轴不拉长。**夹爪既不滤波，也不参与过渡，始终沿用新计划的开合时机。** 下一次 RTC 前缀使用包含这段过渡的执行计划。这是额外执行处理，不能把它的连续性效果归因于原 RTC 公式。

接续前检查整个过渡窗口（含接管漂移余量）内旧执行计划与平滑后新计划的最大分歧。任一臂超过默认 `--rtc-max-position-gap-mm 20` 或 `--rtc-max-rotation-gap-deg 5`，就以 `rtc_plan_disagreement` 保持并暂停，拒绝接管。需要用户检查状态后按空格，重新获取观测恢复；被拒绝的段不能计作成功执行。首次开始或暂停恢复时只核对新计划起始目标与实测反馈的差异，再从该反馈位姿渐入，允许计划后续包含正常任务运动。

需要对照纯模型 RTC 时，显式传 `--rtc-transition-ms 0 --rtc-smoothing-ms 0`；这同时关闭执行滤波、过渡及上述接续分歧检查。它仍保留原 RTC 的时间、数据有效性和迟到暂停检查。

OpenPI 以 `t=1→0` 采样，RTC 的去噪估计为 `x-t*v`，引导项从速度场减去。物理前缀先转为训练时的相机坐标 XYZ/YPR/归一化夹爪开度，再使用训练集的 action quantile normalization，补齐到32维；额外18维不贡献前缀误差。代码位于本项目 `pi05/src/egoverse_pi05/rtc.py`，不修改固定版本的 OpenPI 源码。

先在原服务终端按 Ctrl+C 退出旧服务，再用**已有 JAX 环境**启动 RTC 服务：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
CUDA_VISIBLE_DEVICES=0 ./pi05/.venv/bin/python pi05/scripts/serve_pi05.py \
  --checkpoint-path /home/yesen/pico_ego_probe/EgoVerse/ckpt/egoverse_1107_pick_full_bs8_12000/12000 \
  --openpi-root /home/yesen/pico_ego_probe/EgoVerse/pi05/openpi \
  --num-inference-steps 10 --rtc --rtc-max-guidance 5 \
  --host 127.0.0.1 --port 8000
```

服务在监听端口前预热普通路径和正权重 RTC 路径。等到 `PI05 WARMUP PASSED` 后，在另一个终端启动客户端：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.infer_piper_policy \
  --policy-backend pi05 --server-url ws://127.0.0.1:8000 --server-timeout 10 \
  --training-data /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted/1107_pick \
  --calibration /home/yesen/pico_ego_probe/EgoVerse/calibrations/lrcp_front_480p_identity_extrinsics_001.json \
  --camera-config /home/yesen/pico_ego_probe/EgoVerse/calibrations/uvc_rig.json \
  --task-description 'Pick up the object and place it in the container.' \
  --left-can can0 --right-can can1 \
  --control-frequency 30 --execute-steps 12 \
  --rtc --rtc-delay-ms 200 \
  --rtc-transition-ms 200 --rtc-smoothing-ms 100 \
  --rtc-max-position-gap-mm 20 --rtc-max-rotation-gap-deg 5 \
  --home-on-start true --preview --preview-fps 15 --log --execute
```

`--execute-steps 12` 在 RTC 模式定义**相邻请求起点之间的目标周期**：`12 / 30 = 0.4` 秒。下一次请求从上一次请求的实际发起时刻计时，不再从新段接入后重新数12条命令，因此不会额外累加一次推理预算。若设备回执等开销使某次请求稍晚，只提交一次，并从这次实际请求重新计时；不补发请求、不同时运行多个 RPC。100点预测仍覆盖约0.967秒。

`--rtc-delay-ms 200` 是从请求起点算起的严格结果截止时间，包含读取图像、预处理、网络通信、模型计算和结果检查。后台结果提前到达时，先等待截止时间之后的计划控制 tick，再在实际发送 tick 接入新段；这不会放宽200ms的结果期限。迟到结果不切入，双臂保持并暂停，空格重新观察并推理。这里采用固定预算，论文的延迟历史自适应估计暂未启用。

考虑 SDK 回执阻塞和小幅 tick 漂移，前缀约束会覆盖计划接管 tick 之后一个控制周期，并包含100点模型网格中右侧的插值点。首次发送新目标前会再次检查实际取样时间是否仍处于该硬前缀支持范围；若超出则以 `rtc_hard_coverage_expired` 保持并暂停。旧计划无法覆盖右侧插值点时也拒绝接续，不重复旧末点伪造有效前缀。权重为1仍然是模型内部的引导约束，不代表预测被精确锁定为旧目标。

首次开始或暂停恢复时，保持机械臂并取新观测，不复用暂停前的模型前缀；执行连续性层从本次实测反馈渐入首段。暂停立即请求保持、作废在途结果；退出先保持，再等待有限超时的后台请求退出，最后关闭设备和共享内存。RTC 需要服务器能力握手，不支持的服务会在打开设备前被拒绝。不要同时传 `--async-inference`：那是另一条旧执行模式；RTC 自己已经异步运行，其连续性层由 `--rtc-transition-ms`、`--rtc-smoothing-ms` 控制。去掉客户端 `--rtc` 可回到原同步模式，服务器即使启用 RTC 也兼容普通请求。

预算必须容纳实际 RTC 耗时，不能直接拿原来普通推理的约84ms作为 RTC 耗时。令 `P = execute_steps / 控制频率`、`Δ = 1 / 控制频率`、`D = rtc_delay_ms / 1000`、`T = rtc_transition_ms / 1000`、`g = 最大模型网格间隔`、`H = chunk跨度`，程序要求：

```text
P > D + 2Δ
0 <= T < P
P + D + 2Δ + g + T < H
```

第一项保证上一请求完成接管后才发起下一请求；后两项为完整过渡及右侧插值点保留有效旧计划。当前30Hz、100点配置下，`Δ≈33.33ms`、`g≈9.76ms`、`H≈0.967s`。默认 `P=400ms, D=200ms, T=200ms` 满足要求。若实测需要300ms预算，可使用 `--execute-steps 12 --rtc-delay-ms 300 --rtc-transition-ms 150`；同样400ms请求周期配默认200ms过渡会需要约976ms，超出预测跨度，会被拒绝。旧示例中的6点周期只有200ms，也不满足第一项。运行时还会用实际观测时刻复核剩余时域，不外推旧计划或静默缩短过渡。

超时原因和请求起点保存在 `events.jsonl`，实际提交目标、主机提交时刻及 SDK 回执仍保存在 `commands.jsonl`。`cycles.jsonl` 在首条新目标成功提交后记录结果截止时间 `deadline_monotonic`、计划接管时刻 `activation_walltime`、漂移裕量截止时刻 `hard_until_walltime`、实际硬前缀末节点 `prefix_hard_until_walltime`，以及首条目标取样时刻 `first_target_walltime`、主机提交时刻 `first_send_monotonic`。`switch_walltime` 与首条目标取样时刻一致；`raw_switch_plan_delta` 比较原始新预测与旧执行计划，`switch_plan_delta` 比较加入执行过渡后的新、旧目标，均使用同一绝对时刻。`base_actions` 保留原始base计划，`execution_base_actions` 保存段内滤波后的100点新计划；后者尚不包含过渡采样，实际过渡目标以 `commands.jsonl` 为准。另记 `execution_transition_s`、`execution_smoothing_s`、整窗 `transition_disagreement`，以及模型前缀匹配残差，避免把入口连续误认为模型误差消失。`--no-log` 仍可关闭逐条日志。

每臂 SDK 回执新增 `measured_feedback`，保存确认循环已读取的末端反馈、关节位置、可用时的关节速度、夹爪开度和来源时间戳。末端字段 `ee_pose_xyz_rpy_m_rad` 使用各臂base下的米/弧度及SDK的 `extrinsic_xyz_rpy` 约定。这是缓存的最新反馈，时间戳可能早于当前提交；它不是与发送瞬间原子同步的测量，也不是“已经到达目标”的确认。

RTC 的引导是近似条件约束，不保证每个切换点严格重合。额外的执行连续性层减少命令突变，但可能改变过渡路径，不能修复错误的任务目标或保证机械臂物理跟踪。离线滤波、融合对照和单元测试不能承诺真机平稳性或任务成功率；应结合整段命令、实际反馈、拒绝接管次数及任务效果评估。

### 不连接真机的测试

所有命令复用已有环境，不安装依赖。CPU 单元测试覆盖协议、非单位外参、夹爪单位、100点时间轴、旋转、延迟变化、暂停恢复、过期结果丢弃和异步日志。`test_policy_rtc_timing.py` 用虚拟时钟模拟20ms回执和1ms唤醒延迟；`test_policy_continuity.py` 检查入口/出口速度、SO(3)短弧、夹爪不变、滤波与批量采样等价性；`test_policy_rtc_continuity.py` 验证连续性层接入RTC、分歧拒绝和恢复：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_pi05_rtc_server.py tests/unit/test_pi05_rtc_remote.py \
  tests/unit/test_policy_rtc.py tests/unit/test_policy_rtc_timing.py \
  tests/unit/test_policy_continuity.py tests/unit/test_policy_rtc_continuity.py \
  tests/unit/test_live_policy.py \
  tests/unit/test_live_policy_devices.py tests/unit/test_pi05_remote.py

./pi05/.venv/bin/python pi05/scripts/check_pi05_rtc.py --cpu-only

CUDA_VISIBLE_DEVICES=0 ./pi05/.venv/bin/python pi05/scripts/check_pi05_rtc.py \
  --checkpoint-path /home/yesen/pico_ego_probe/EgoVerse/ckpt/egoverse_1107_pick_full_bs8_12000/12000 \
  --manifest /home/yesen/pico_ego_probe/EgoVerse/data/pi05/1107_pick/manifest.json \
  --openpi-root /home/yesen/pico_ego_probe/EgoVerse/pi05/openpi \
  --output diagnostics/pi05_rtc_new_run/gpu_smoke.json
```

最后一条使用真实训练图像和 state，验证实际 checkpoint 的 RTC 路径、前缀归一化、零权重与原模型同噪声一致、正权重重复结果，以及预热耗时。输出 JSON 和 NPZ；输出文件已存在会拒绝覆盖。这属于实现验证，不能替代闭环任务效果测试。

2026-11-08 实测结果保存在 `diagnostics/pi05_rtc_20261108/gpu_smoke_final.json`：直接调用新增 RTC 采样器、以零权重和相同噪声与原采样器比较，最大输出差为0；正权重结果有限且可复现。测试把一个训练输入对应的旧前缀 X 平移5mm，引导后的该项平均误差为3.362mm，加权归一化前缀误差下降约31.3%。这是一项人为构造的前缀约束检查，不是真机 chunk 跳变降低31.3%。单次预热调用约70ms（普通）／110ms（RTC），当时有另一模型服务驻留 GPU，只作为初始预算参考。

RTC 调度按实际主机提交时刻安排下一次下发，避免微小唤醒误差持续累积；模型轨迹仍按绝对时间取样，不拉长预测时间轴，也不补发过期点。单次明显停顿仍会暂停。测试包含1000次以上提交时每次睡眠多1ms的情形，以及变化的健康检查开销。这里控制的是主机提交节奏，CAN worker 的实际 SDK 发送仍有自身抖动，可在 `commands.jsonl` 的设备回执中检查。

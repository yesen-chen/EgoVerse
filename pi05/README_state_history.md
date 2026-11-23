# Pi0.5：按实际时间选择 TCP 与夹爪历史

当前 state 仍为实际记录的双臂反馈 14 维。新增 `observation.state_history` 为 `H×14`，每帧包含左右 TCP 和夹爪；动作定义及插值到 100 点的流程保持不变。

历史偏移在导出 manifest 时指定，以当前 observation 时间为零点。可以用一帧、两帧或更多帧，也可以用非控制周期整数倍，例如 `-90 -47 -21` 毫秒。偏移必须严格为负、从最早到最近排列且不重复。每个 checkpoint 绑定一组固定偏移；变更偏移或帧数要重新训练，不能部署时随意改动。

## 时间与坐标约定

- Piper 历史来自原始高频反馈 `robot0_state.npz`、`robot1_state.npz`，使用各自时间戳，位置和夹爪线性插值，旋转用 SLERP。这里的时间仍是采集系统记录的主机反馈时间，不宣称为硬件曝光或 CAN 精确同步时间。
- 当前 state 完全保留原记录。历史插值的两个端点都必须在当次实际读到的缓冲范围内；离线使用 `source.left/right.robot_timestamp` 限制可用数据，不能利用后来到达的反馈。
- 历史不足的训练行被明确排除，manifest 保存 `eligible_indices`。不会补成静止状态，也不会跨 episode 取历史。非常接近零的偏移可能因反馈尚未到达而没有有效样本。
- 在线从同一次反馈 SHM 快照生成当前 state 与历史，启动时等待缓冲覆盖所需时段；超过原默认容量的历史会自动增大反馈缓冲。绝不使用前几次模型请求的 state 代替高频反馈。
- Piper 每个历史位姿都使用对应臂的外参转换到相机系。Ego 从已转换数据的世界系手部位姿取历史，再统一变换到**当前样本**相机系；其时间分辨率受源数据限制，不会凭空获得 100 Hz 信息。
- 夹爪包含在历史中。Ego `zero` 模式历史夹爪与当前夹爪一样，在归一化后置零。
- Pi0.5 使用已有离散 state token 方式编码额外历史，并共用当前 state 的归一化尺度。物理动作维度仍为 14、模型 padding 仍为 32。足够小的运动可能落入同一离散区间，因此历史输入不保证消除抖动。

## 已有 1107_pick 数据：补回填并导出新 manifest

环境由用户管理。以下命令使用已有环境，不安装依赖。补回填仅添加独立历史文件，不修改原有 Zarr 数组、图像或根 metadata；会核对原始数据与转换数据的对应关系。重复同一回填是幂等的，来源不一致会报错。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

python -m egomimic.scripts.backfill_piper_state_history \
  --source recordings/1107_pick \
  --converted-root data/piper_converted/1107_pick

python -m egomimic.scripts.export_piper_pi05 \
  --input data/piper_converted/1107_pick \
  --output data/pi05/1107_pick_history/manifest.json \
  --source-fps 30 --source-horizon 30 --action-stride 1 \
  --validation-ratio 0.1 --seed 42 \
  --state-history-offsets-ms -66.6666666667 -33.3333333333
```

也可以传 `--state-history-offsets-ms -50 -25` 或 `-90 -47 -21`。不传该参数即为原来的无历史输入。新的原始数据转换会自动保存可用的高频反馈；在 `piper_pico_to_zarr` 的 `--pi05-manifest` 导出命令中，加 `--pi05-state-history-offsets-ms` 即可一并启用。

Ego 导出入口 `egomimic.scripts.export_pico_ego_pi05` 使用同名 `--state-history-offsets-ms`。混合前两个 manifest 的历史约定必须一致；详见原混合训练文档。

## 训练前检查与训练

```bash
PI05_PYTHON=/home/yesen/pico_ego_probe/EgoVerse/pi05/.venv/bin/python
PI05_OPENPI=/home/yesen/pico_ego_probe/EgoVerse/pi05/openpi
PI05_PARAMS=/home/yesen/cys/avfrl_real/pi05/.cache/openpi/pi05_base/params
PI05_TOKENIZER=/home/yesen/cys/avfrl_real/pi05/.cache/openpi/big_vision/paligemma_tokenizer.model
PI05_MANIFEST=/home/yesen/pico_ego_probe/EgoVerse/data/pi05/1107_pick_history/manifest.json
PI05_STATS=/home/yesen/pico_ego_probe/EgoVerse/data/pi05/1107_pick_history/stats

# 不连模型/网络/设备：对比训练输入与实际推理客户端发出的输入。
python -m tests.integration.check_piper_history_parity \
  --manifest "$PI05_MANIFEST" --samples 24

"$PI05_PYTHON" pi05/scripts/compute_pi05_stats.py \
  --manifest "$PI05_MANIFEST" --output-dir "$PI05_STATS" \
  --openpi-root "$PI05_OPENPI"

"$PI05_PYTHON" pi05/scripts/train_pi05.py \
  --manifest "$PI05_MANIFEST" --stats-dir "$PI05_STATS" \
  --tokenizer-path "$PI05_TOKENIZER" --params-path "$PI05_PARAMS" \
  --openpi-root "$PI05_OPENPI" --exp-name pick_pi05_history \
  --batch-size 1 --max-token-len 256 --dry-run
```

无历史模型的默认 token 长度为 200，启用历史后默认为 256；较多历史帧或长任务文本可能需要 `--max-token-len 384` 等更大预算。历史模式实际 token 超限会报错，不能静默截断历史。先跑 dry-run，再将原训练命令的 manifest、stats、exp-name 换为新实验路径，并保持 dry-run 验证过的 token 预算。全量/LoRA、学习率与动作窗口参数沿用原方案。

部署服务仍用原 `serve_pi05.py`，客户端仍用 `infer_piper_policy`。双方从 checkpoint 读取历史偏移和 token 配置，自动启用 SHM 历史；客户端不提供覆盖训练历史的开关。旧 checkpoint 不要求历史。历史 checkpoint 不能通过旧的无历史训练状态直接 `--resume`；应作为新实验初始化。

## 测试入口

重要环节分别有测试，均不需要连接真机：

| 文件 | 主要验证 |
| --- | --- |
| `tests/unit/test_piper_state_history.py` | 时间采样、旋转插值、未来隔离、可搬迁反馈文件及校验 |
| `tests/unit/test_piper_pico_to_zarr.py` | 转换保留历史、同源回填、CLI 转发及兼容 |
| `tests/unit/test_pi05_state_history_data.py` | 任意窗口、14 维历史、覆盖过滤、Ego 当前相机系、混合约定 |
| `tests/unit/test_state_history_live.py` | 同次 SHM 快照、RPC 字段、坐标/夹爪单位、旧模型拒绝误配 |
| `tests/unit/test_state_history_devices.py` | 长历史缓冲环绕、启动等待、超时、取消及旧模式 |
| `tests/unit/test_state_history_parity.py` | 训练与在线输入逐项对齐、故意破坏历史时能检出 |
| `tests/unit/test_pi05_history_assets.py` | checkpoint 历史绑定、恢复、服务输入形状及模型配置 |
| `tests/integration/check_pi05_state_history.py` | 真实 OpenPI/tokenizer、归一化边界、历史 token、RTC 前缀 |

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_piper_state_history.py \
  tests/unit/test_pi05_state_history_data.py \
  tests/unit/test_state_history_live.py \
  tests/unit/test_state_history_devices.py \
  tests/unit/test_state_history_parity.py \
  tests/unit/test_pi05_history_assets.py

"$PI05_PYTHON" tests/integration/check_pi05_state_history.py \
  --openpi-root "$PI05_OPENPI" --tokenizer-path "$PI05_TOKENIZER"
```

实际预训练权重 LoRA 测试沿用 `pi05/scripts/check_pi05_lora.py`：传上面的历史 manifest、stats，先 `--inspect`，再 `--steps 2`，最后 `--steps 4 --resume`。它检查 LoRA/优化器确实更新、冻结权重样本不变、保存与续训一致。完整训练→续训→推理回归也有显式历史案例：

```bash
export EGOVERSE_TEST_PI05_PYTHON="$PI05_PYTHON"
export EGOVERSE_TEST_PI05_OPENPI_ROOT="$PI05_OPENPI"
export EGOVERSE_TEST_PI05_MANIFEST="$PI05_MANIFEST"
export EGOVERSE_TEST_PI05_STATS_DIR="$PI05_STATS"
export EGOVERSE_TEST_PI05_HISTORY_MANIFEST="$PI05_MANIFEST"
export EGOVERSE_TEST_PI05_HISTORY_STATS_DIR="$PI05_STATS"
export EGOVERSE_TEST_PI05_TOKENIZER="$PI05_TOKENIZER"
export EGOVERSE_TEST_PI05_PARAMS="$PI05_PARAMS"
export EGOVERSE_TEST_PI05_HISTORY_MAX_TOKEN_LEN=256
CUDA_VISIBLE_DEVICES=0 python -m pytest --integration -m gpu -q \
  tests/integration/test_pi05_jax_lora.py -k history
```

建议对照实验固定图像、动作定义、数据划分和优化器设置，仅改变历史输入。若历史覆盖过滤删除了部分行，比较时使用相同有效 observation 行，不能将不同数据子集的差异全归因于历史。训练结束后再分别量化 chunk 内部变化、接管残差和静止重复预测；这些测试通过只说明管线符合约定，不代表模型已经改善抖动。

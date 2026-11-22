# PICO Ego 与 Piper 的 π0.5 混合训练

这里接入已经转换、去畸变、按时间戳对齐的 Ego Zarr，不重新处理视频，也不修改源时间戳。原 Piper 单独训练命令和模型服务接口继续有效。

## 时间与动作约定

| 域 | 本轮配置 | 源首末时间跨度 | 监督来源 |
|---|---|---|---|
| Piper | 30 Hz，50 帧，stride=1 →100 点 | 49/30 = 1.633333 s | 各臂 command |
| PICO Ego | 30 Hz，30 帧，stride=1 →100 点 | 29/30 = 0.966667 s | 世界系未来 observed 虚拟 TCP |

两者窗口均包含当前帧。100 点对齐的是轨迹的相对进度，不表示100 Hz。两种源时间跨度比例为49/29≈1.69；这不能保证机器人完成任意动作都恰好慢1.69倍。模型部署按 Piper 时间约定执行，不将 Ego 的约0.97秒误当机器人窗口时长。

原 EgoVerse Human 默认是H30/stride3，Eva是H45/stride1，共同输出100点；本轮H30/stride1与H50/stride1是显式可配置的本地方案，并非声称完全照搬原始超参数。

- Piper当前state使用实测末端；未来action使用command。均从各自base变换到该段固定相机坐标系。
- Ego当前state和整个未来action均变换到**当前观测时刻**的光学相机系；未来点不会使用未来相机位姿。
- 坐标为米，旋转为intrinsic ZYX弧度，夹爪为固定比例宽度；模型输入每臂xyz+ypr+gripper，共14维，然后补零到32维。
- Ego `zero`为默认：归一化之后将夹爪维6、13置零，再进行state tokenization；`width`使用已经保存的固定比例宽度，不重复除最大开度。
- Ego只有front_1；两路腕部图像以零占位、mask=False。Piper三路图像mask=True。全部使用同一OpenPI缩放/补边流程，没有再次去畸变。

## 1. 从 Ego Zarr 导出清单

使用新目录；已有清单和统计不会覆盖。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

python -m egomimic.scripts.export_pico_ego_pi05 \
  --input diagnostics/ego_uvc_preprocess_20261115/converted_final \
  --output data/pi05/1115_ego_h30_zero_v1/manifest.json \
  --source-fps 30 --source-horizon 30 --action-stride 1 \
  --gripper-mode zero --validation-ratio 0.1 --seed 42
```

该入口要求当前PICO Ego转换格式、`piper_x`轴向和完整图像几何元数据。它按原始tracking文件SHA划分训练/验证，同一原始录像切出的片段不能跨集合。不会将任意旧版Human/Eva数据猜成该格式。

默认两种人工结尾都**禁止末尾重复**，仅有完整H帧的起点可训练；先判断H帧是否齐全，再应用stride。不会删除尾部行。可分别改变：

- `--allow-repeat-after-trim`：允许首尾裁剪形成的结尾重复。
- `--allow-repeat-after-inactivity`：允许失活、tracking/video gap形成的结尾重复。

自然`source_end`和没有预处理元数据的`legacy`结尾沿用自然结尾重复规则；边界原因和实际规则随每条episode保存。严格窗口短于H的片段会明确报错，不会静默减少数据。

更改窗口、stride、边界开关或gripper模式后，应生成新清单，并重新计算统计。

## 2. 合并两种清单

```bash
python -m egomimic.scripts.export_piper_ego_pi05 \
  --piper-manifest data/pi05/1107_pick_h50/manifest.json \
  --ego-manifest data/pi05/1115_ego_h30_zero_v1/manifest.json \
  --output data/pi05/piper50_ego30_zero_v1/manifest.json \
  --piper-weight 1 --ego-weight 1 --seed 42
```

默认按1:1采样，支持正整数权重。每个有限采样周期覆盖两域全部合法训练起点，较小的域循环采样；上游DataLoader再打乱顺序。1:1不是在每个随机batch里强制各占一半。验证集不重复采样。

混合清单引用两份子清单和Zarr，移动时需保留其相对目录关系。子清单、数值或元数据变化会拒绝加载。JPEG不逐字节哈希。

## 3. 独立统计与 CPU 输入检查

统计使用各域**不重复的合法训练起点**，改变混合权重不会改变统计。Ego验证片段不参与；Piper沿用其子清单的训练/验证划分，用户当前`1107_pick_h50`为validation_ratio=0，没有Piper留出集。

```bash
pi05/.venv/bin/python pi05/scripts/compute_pi05_stats.py \
  --manifest data/pi05/piper50_ego30_zero_v1/manifest.json \
  --output-dir data/pi05/piper50_ego30_zero_v1/stats \
  --openpi-root pi05/openpi

pi05/.venv/bin/python pi05/scripts/train_pi05.py \
  --manifest data/pi05/piper50_ego30_zero_v1/manifest.json \
  --stats-dir data/pi05/piper50_ego30_zero_v1/stats \
  --tokenizer-path /home/yesen/cys/avfrl_real/pi05/.cache/openpi/big_vision/paligemma_tokenizer.model \
  --params-path /home/yesen/cys/avfrl_real/pi05/.cache/openpi/pi05_base/params \
  --openpi-root pi05/openpi --exp-name mixed_input_check \
  --batch-size 2 --dry-run
```

输出`norm_stats.json`为Piper，`ego_norm_stats.json`为Ego。使用各自OpenPI q01/q99归一化；混合输入不会再次全局归一化。`--max-windows`仍只用于测试，正式统计不传。`--dry-run`不加载完整模型或启动优化器。

正式训练沿用主README的`train_pi05.py`命令，只换成混合manifest和stats目录。混合checkpoint同时保存两份统计和两份完整时间/数据约定，恢复时检查其一致性。服务端只向机器人暴露Piper几何、时间及反归一化统计，不接受Ego轨迹作为机器人回放。

`check_pi05_inference.py`对混合模型做机器人离线推理检查时，应传原Piper子清单，例如`data/pi05/1107_pick_h50/manifest.json`。

## 4. 验收脚本

```bash
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
EGOVERSE_TEST_PI05_PYTHON="$PWD/pi05/.venv/bin/python" \
python -m pytest -q \
  tests/unit/test_pi05_ego_data.py \
  tests/unit/test_pi05_training_data.py \
  tests/unit/test_pi05_mixed_inputs.py \
  tests/unit/test_pi05_mixed_assets.py

python tests/integration/check_pi05_mixed_recordings.py \
  --manifest data/pi05/piper50_ego30_zero_v1/manifest.json \
  --stats-dir data/pi05/piper50_ego30_zero_v1/stats \
  --output diagnostics/my_mixed_acceptance.json
```

新测试覆盖当前相机坐标系、SLERP、时间跨度、四种边界开关组合、来源分组、图像mask、zero/width归一化、不同采样权重、原Piper兼容性、便携checkpoint和恢复拒绝错误配置。

实际数据验收产物位于`diagnostics/pi05_ego_mixed_20261122/`：48段Ego加44段Piper，Ego6145行保留、4753合法起点（训练4197／验证556）；Piper22786训练起点。两套完整统计已经生成。`acceptance.json`记录几何检查，`cpu_input_check.txt`记录真实混合batch检查。验收不等同于完成预训练权重的GPU优化步骤或任务成功率评估。

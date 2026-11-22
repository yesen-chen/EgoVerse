# 外置 UVC 联合录像预处理与 Ego Zarr 转换

入口：`python -m egomimic.scripts.pico_uvc_ego_to_zarr`。
旧 `pico_ego_to_zarr` 继续用于 PICO 内置相机的 `CameraRecord_*.mp4` / `trackingData_*.txt`，不要把新格式改名后送入旧入口。

## 默认处理

- 每条原始录像开头、结尾各裁 **1 秒**，可独立配置。
- 双手任一失活或关节位置无效，就在**原始高频追踪时间轴**切开。即使失活落在两个 30 Hz 时刻之间也不能跨过；这里不使用旧转换器的短时保持。
- 头部缺测、超过 50 ms 的追踪/视频时间间隙同样切开，不连接前后片段。
- 切分后首尾保留行的时间跨度不足 **2 秒**则过滤，可配置。这是 `(N-1)/fps`，不是 `N/fps`。
- 保留行处于统一 30 Hz 时间格。默认位姿插值，`--pose-sampling nearest` 可选择最近原始姿态。最近图像按逐帧 receive UTC 选择；真实 MP4 PTS 和帧 ID 另存，绝不把帧号除以帧率当时间戳。
- 使用冻结拟合的头、手偏移各一次。原始 UTC 保持整数纳秒，SDK 的零头部源时间戳不参与对齐。
- 使用拟合文件内冻结的原始 K/D，复用真机图像处理：去畸变（默认 alpha=0、不裁 ROI）→缩放→JPEG。输出有效 K、D=0 和完整原图参数。
- 世界系手部虚拟 TCP、世界系光学相机轨迹、宽度、源支持样本时间均保留；训练时统一变到当前相机系。默认 Piper-X 轴约定。

原始文件不改动。每个原始录制对应一个独立输出目录，其中是若干 `episode_*.zarr` 和 `preprocessing_report.json`。没有可用片段的录像也保留报告；不制造空 Zarr。已有目录拒绝覆盖，单条录制全部成功后才发布输出。

## 输入、预检查、转换

每个 session 应包含 `ego.mp4`、`frames.csv`、`tracking.jsonl`、`session.json`、`joint_session.json`、`tracking_summary.json`。输入可以是一条 session，也可以是直接包含多条 session 的目录。

拟合文件使用 `seeded_external_uvc_spacetime_v1`：`parameters.R_CH/t_CH_m/delta_head_ms/delta_hand_ms`，以及 `intrinsics.raw_K/raw_D/image_size/mirror=false`。外参是 `T_C_H`，头、手查询为图像时钟上的采样时刻加对应拟合偏移。不能传入旧的双目拟合文件，也不应再次扣除相机测得的延迟。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

# 只写裁剪、失活、短片段过滤和时间对应报告；不写 Zarr。
python -m egomimic.scripts.pico_uvc_ego_to_zarr \
  --input ../pico_uvc_probe/joint_captures_20261115/sessions \
  --fit ../uvc_head_calibration_20261115/fit/fit.json \
  --output-dir data/pico_ego_review/1115 \
  --preprocess-only

# 使用另一个输出目录执行完整转换。
python -m egomimic.scripts.pico_uvc_ego_to_zarr \
  --input ../pico_uvc_probe/joint_captures_20261115/sessions \
  --fit ../uvc_head_calibration_20261115/fit/fit.json \
  --output-dir data/pico_ego_converted/1115 \
  --trim-start-s 1 --trim-end-s 1 --min-segment-s 2 \
  --fps 30 --width 640 --height 480 \
  --pose-sampling interpolate --axis-convention piper_x \
  --task-name pick_place --task-description 'Bimanual pick and place'
```

首尾 1 秒是固定规则，**不是 UI 点击识别器**。多次点击、伸手操作 UI 若超出 1 秒，需要复查并为对应录像加大裁剪量。`--trim-overrides FILE.json` 支持逐条覆盖，例如：

```json
{
  "joint_20261115_151302_029_dd758ed2": {
    "trim_start_s": 1.0,
    "trim_end_s": 2.0
  }
}
```

报告记录所有候选区间、保留/过滤原因和时间边界。裁剪以第一/最后一个实际保存视频帧的 receive UTC 为参考，并要求采样时刻与选中图像都在保留范围内。选中图像也不能跨到另一侧的失活区间。

## 两个独立的训练开关

在 `recordings/piper_pico.yaml` 中：

```yaml
ego_require_full_window_after_trim: true
ego_require_full_window_after_inactivity: true
```

两个默认都是 **true（只采完整未来窗口，禁止末尾重复）**。这对应讨论里“默认不启用末尾重复”。设为 `false` 才允许对应片段末尾补重复动作。

| 片段的结束原因 | 使用的设置 |
| --- | --- |
| `ui_trim`：结束 UI 裁剪 | `ego_require_full_window_after_trim` |
| `inactivity` / `tracking_gap` / `video_gap` | `ego_require_full_window_after_inactivity` |
| 完整源结束 `source_end` 或没有预处理元数据的旧数据 | 保留原有末尾重复行为 |

按**结束原因**选择，而不是“这段是否曾从失活处开始”。例如中间失活后恢复，到 Stop 前裁剪结束的后一段，使用首尾裁剪开关。

可以向既有 HPT 本地 Ego / 混合训练命令添加以下覆盖项：

```text
recordings.ego_require_full_window_after_trim=false
recordings.ego_require_full_window_after_inactivity=false
```

无需重新转换 Zarr。保留 N 行、源窗口 H 行时，严格策略有 `N-H+1` 个训练起点；末尾 H−1 行仍是前面起点的真实未来目标，**不能从文件删除**。H 是 stride 前的完整源窗口，模型输出插值点数不会改变这些规则。片段若不足 H 行，会明确报错并列出文件，需要增大预处理最短时长或减小 H。

训练、验证、归一化统计使用同样的合法起点。策略保存到 checkpoint 和统计指纹；改变策略后应重算统计。旧 Ego checkpoint 没有这项契约时不能无提示续训。Piper 真机与现有仅用于 Piper 的 π0.5 导出/训练路径保持原样；这个入口并没有新增 Ego→π0.5 混合训练适配。

不同切片按原始 `tracking.jsonl` 哈希分组划分训练/验证，不会把同一录制的前后片段分到两边。

## 投影与独立验收

```bash
python -m egomimic.scripts.data_visualization.visualize_pico_ego_zarr \
  --input data/pico_ego_converted/1115/某条session \
  --output-dir data/pico_ego_review/1115_projection/某条session

python tests/integration/check_pico_uvc_preprocessing.py \
  --raw-root ../pico_uvc_probe/joint_captures_20261115/sessions \
  --converted-root data/pico_ego_converted/1115 \
  --fit ../uvc_head_calibration_20261115/fit/fit.json \
  --report data/pico_ego_review/1115_acceptance.json
```

独立验收检查源数据哈希、每行 UTC/PTS/支持样本、偏移次数、raw 高频失活与间隙、相机/TCP 变换、首尾像素处理、全部四种窗口开关组合，以及训练/统计一致性。报告中的 H=50 是验收参数，未覆盖训练配置里的实验窗口。

投影检查读取**转换后的 JPEG 和 pose**，不会重新拟合。数学链路一致只能证明实现与给定标定一致，不能替代手部跟踪精度或标定的人工验证。

单元/回归测试：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q \
  tests/unit/test_ego_joint_input.py \
  tests/unit/test_ego_preprocessing.py \
  tests/unit/test_pico_uvc_ego_to_zarr.py \
  tests/unit/test_ego_boundary_policy.py \
  tests/unit/test_pico_ego_to_zarr.py \
  tests/unit/test_visualize_pico_ego_zarr.py
```

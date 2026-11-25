# PiPER 实时同步策略推理

`python -m egomimic.scripts.infer_piper_policy` 使用三台 UVC 相机、双臂 SDK 实际反馈和本地 checkpoint 进行同步推理，通过终端键盘开始、暂停和退出。**不需要 PICO 头显、手柄或 `runService.sh`。** 加 `--offline-eval` 则使用录制数据完成离线误差验证，不打开硬件；另有 [离线策略回放](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_policy_inference.md) 用于虚拟时间调度检查。

Pi0.5 JAX 远程服务支持可选 `--rtc`：模型前缀引导与后台推理配合执行，保留100点训练动作与标定约定。服务端也要启用 `--rtc`，客户端建议从 `--execute-steps 12 --rtc-delay-ms 200` 开始。RTC 的12点表示相邻请求起点相隔目标 `12/30=0.4` 秒，下一请求从上次实际请求时刻计时。完整命令及不连接真机的测试见 [Pi0.5 README 的 RTC 部分](/home/yesen/pico_ego_probe/EgoVerse/pi05/README.md#rtc在模型内部约束相邻动作段)。以下未启用 RTC 的同步行为保持不变。

启用 RTC 后，客户端默认另外使用执行连续性层：`--rtc-transition-ms 200` 用五次 smoothstep 在同一绝对时刻的旧、新轨迹之间衔接位置和 SO(3) 短弧姿态；`--rtc-smoothing-ms 100` 对完整新预测做居中位姿滤波，当前100点网格采用SG11/3阶，实际窗口约97.6ms。**夹爪既不滤波，也不blend，保持新计划原本的开合时机。** 首次开始或暂停恢复时，从本次实测反馈渐入。过渡结束后使用新计划，预测时间轴不延长；后续模型RTC前缀来自包含过渡的执行计划。这是原RTC之外的执行层，不表示模型引导本身已保证连续。

接续前，程序检查完整过渡窗口（含接管漂移余量）内旧执行计划与平滑新计划的分歧。任一臂最大位置差超过默认20mm或SO(3)旋转差超过5°，以 `rtc_plan_disagreement` 保持并暂停，拒绝接管；检查状态后按空格重新观察恢复。首次/恢复只核对起始预测与实测反馈的差异，后续正常任务运动不按恒定反馈整窗拒绝。被拒绝的段不能算作成功执行。阈值分别由 `--rtc-max-position-gap-mm`、`--rtc-max-rotation-gap-deg` 配置。传 `--rtc-transition-ms 0 --rtc-smoothing-ms 0` 可对照纯RTC，同时关闭这层滤波、过渡及分歧检查；原有时间和数据有效性检查仍保留。

RTC 的结果截止时间与接管时刻分开：200ms是读取观测、推理和结果检查的严格预算；合格结果在截止时间之后的计划控制 tick、实际准备发送命令时接入。硬前缀覆盖该 tick、一个控制周期的漂移裕量及右侧100点模型插值节点。首条新目标若超出硬前缀支持范围，就保持并暂停，不切入软尾。旧段 horizon 之外不会重复末点作为约束。实际取样时刻与主机提交时刻分别记录为 `first_target_walltime`、`first_send_monotonic`；`switch_plan_delta` 比较同一取样时刻的新旧计划。

RTC 配置要求 `P > D + 2Δ`、`0 <= T < P`，且 `P + D + 2Δ + g + T < H`：其中 `P=execute_steps/频率`、`D=rtc_delay_ms/1000`、`T=rtc_transition_ms/1000`、`Δ=1/频率`、`g` 为最大模型点间隔、`H` 为预测跨度。当前30Hz、100点对应 `g≈9.76ms`、`H≈0.967s`。需要300ms预算时可用 `--execute-steps 12 --rtc-delay-ms 300 --rtc-transition-ms 150`；保留默认200ms过渡则总时域约976ms，超出预测跨度，会被拒绝。旧的6点示例也不满足请求周期要求。运行时复核旧、新计划剩余时域，不外推旧段或静默缩短过渡。引导上限5、flow步数10和100点输出保持不变；应同时核对模型残差、整段执行轨迹和任务效果。

## 当前训练的完整命令

在交互终端运行，使用已经保存完成的 epoch 权重。相机端口、采集参数和内参文件从 `uvc_rig.json` 读取；左右 CAN 默认 `can0`、`can1`。先退出占用相机或控制双臂的采集、遥操作、轨迹回放程序。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

POLICY_CHECKPOINT=/home/yesen/pico_ego_probe/EgoVerse/logs/piper_pico/1107_pick_identity_extrinsics_2026-11-07_19-42-43/checkpoints/epoch_epoch=49.ckpt

python -m egomimic.scripts.infer_piper_policy \
  --checkpoint "$POLICY_CHECKPOINT" \
  --calibration /home/yesen/pico_ego_probe/EgoVerse/calibrations/lrcp_front_480p_identity_extrinsics_001.json \
  --training-data /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted/1107_pick \
  --camera-config /home/yesen/pico_ego_probe/EgoVerse/calibrations/uvc_rig.json \
  --device cuda:0 \
  --left-can can0 --right-can can1 \
  --control-frequency 30 --execute-steps 30 --inference-steps 10
```

以上是**只观察模式**：打开相机、读取 CAN 实际反馈，开始后预测并记录候选动作，不使能机械臂，不下发机械臂/夹爪目标，也不会回 home。它需要硬件在线；若要完全离线用训练图像和 state 验证预测误差，请使用下节的 `--offline-eval`。

实际执行使用：

```bash
python -m egomimic.scripts.infer_piper_policy \
  --checkpoint "$POLICY_CHECKPOINT" \
  --calibration /home/yesen/pico_ego_probe/EgoVerse/calibrations/lrcp_front_480p_identity_extrinsics_001.json \
  --training-data /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted/1107_pick \
  --camera-config /home/yesen/pico_ego_probe/EgoVerse/calibrations/uvc_rig.json \
  --device cuda:0 \
  --left-can can0 --right-can can1 \
  --control-frequency 30 --execute-steps 30 --inference-steps 10 \
  --execute --home-on-start true
```

`--home-on-start true` 复用采集脚本的双臂 home 关节位置，**同时打开夹爪**，等待反馈到位后进入暂停状态。默认 `false`，保持当前位置；推理启动使能时也保持实测夹爪开度，不主动闭合。只观察模式搭配 `true` 会在打开设备前报错。Home 等待超时默认 20 秒，可用 `--home-timeout` 调整。

运行日志默认开启。可在上述命令末尾加 `--no-log`，关闭逐动作、chunk 和事件日志；显式开启用 `--log`。关闭后仍显示终端状态、暂停和错误，并保留启动参数、模型/标定配置和结束摘要。这两个开关不影响 `--offline-eval` 的评估报告。

开启时，`commands.jsonl`、`cycles.jsonl`、`events.jsonl` 由独立进程进行 JSON 编码和写盘；`requests.jsonl` 用另一个独立进程记录每次预测的输入数值、帧引用和原始输出。控制与请求日志使用独立的有界队列，不等待队列空位或磁盘写入。队列满时丢弃新日志，后台写入失败时停止接受日志，不会仅因日志故障中断动作。退出控制、清理设备后排空后台日志，摘要中的 `logging` 和 `request_logging` 分别保存数量、错误和完整性。`written` 不代表断电持久化。

要保存三路全部画面、精确请求图像及连续双臂反馈，在推理命令末尾加 `--log --record-images all --record-feedback`。图像和反馈读取、编码及写盘在独立后台进程进行，默认图像/高频反馈关闭。无损模式、仅请求模式、目录结构、故障边界和无硬件压力测试见 [推理记录说明](README_policy_recording.md)。

## 离线验证：训练图像、state → 同一推理入口 → 动作偏差

不连接相机、CAN 或头显，不需要交互终端。默认用 CPU、最多 4 个计算线程，避免占用训练的 GPU。四个路径仍传入相同文件，用于检查 checkpoint、标定和数据处理配置。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

python -m egomimic.scripts.infer_piper_policy \
  --offline-eval \
  --checkpoint /home/yesen/pico_ego_probe/EgoVerse/logs/piper_pico/1107_pick_identity_extrinsics_2026-11-07_19-42-43/checkpoints/epoch_epoch=49.ckpt \
  --calibration /home/yesen/pico_ego_probe/EgoVerse/calibrations/lrcp_front_480p_identity_extrinsics_001.json \
  --training-data /home/yesen/pico_ego_probe/EgoVerse/data/piper_converted/1107_pick \
  --camera-config /home/yesen/pico_ego_probe/EgoVerse/calibrations/uvc_rig.json \
  --device cpu \
  --control-frequency 30 --execute-steps 30 --inference-steps 10 \
  --eval-samples-per-episode 8 --eval-parity-samples 4 --eval-plots 6
```

默认每集均匀抽取 8 个窗口，覆盖所选目录中所有 episode；只选有完整未来动作的索引，排除末尾重复补齐标签。`--eval-max-episodes 2 --eval-samples-per-episode 2` 可用于快速检查。`--eval-samples-per-episode` 设置为大于最长 episode 的帧数，可遍历每集所有完整窗口。`--eval-seed` 默认 42，为每集每帧固定 FM 随机种子；每窗口只预测一次，不挑选多次预测中的最佳结果。运行时禁止同时加 `--execute` 或 `--home-on-start true`。

程序调用与实时入口相同的 checkpoint 加载、`PiperPolicy.predict`、坐标转换、夹爪限制和动作时间重采样。输入只含三张训练 RGB 图像和实际 `obs` state，标签不送入网络。具体核对：

1. 每个样本的推理输入与真实 `PiperDataset` 加载器输出逐项比较，覆盖图像通道、像素范围、state 坐标和单位。
2. 将训练标签逆变换回各臂 base，与独立读取的原始 `cmd_ee_pose` / `cmd_gripper` 插值结果比较，检查外参方向、左右臂、单位和时间轴。
3. 默认前 4 个样本固定相同随机种子，分别通过推理适配器输入和训练加载器输入预测，比较模型输出。
4. 比较原始模型全轨迹、首动作、重采样后执行前缀与 command 标签。位置为欧氏距离（mm），旋转为 SO(3) 角距离（°，避免欧拉角跨 ±π 问题），夹爪为开度绝对误差（mm）。另计算“保持当前实测状态”基线与单独的重采样误差。

默认报告位于 `diagnostics/piper_offline_eval_<UTC>_<id>/`；`--output-dir` 可指定新的目录，不覆盖既有结果：

| 文件 | 内容 |
| --- | --- |
| `report.md`、`summary.json` | 中文摘要、均值/中位数/P95/RMSE、首动作和运动/近静止分组 |
| `error_by_horizon.png` | 位置、旋转、夹爪误差随预测时间的变化，并对照保持状态基线 |
| `plots/*.png` | 抽样三路输入图像、双臂 base 下预测/标签轨迹、逐时刻误差 |
| `samples/*.npz` | 原始 camera/base 预测、训练标签、执行前缀和对应时间轴 |
| `samples.jsonl` | 每个窗口索引、种子、误差和输入检查 |
| `manifest.json`、`arguments.json`、`events.jsonl` | checkpoint 哈希、标定与数据配置、参数及夹爪截断记录 |

`pipeline_checks=passed` 表示上述一致性检查通过，不表示模型精度或任务成功率达标。训练集成绩衡量拟合情况，需要分别关注运动机械臂和近静止机械臂；后者的保持状态基线可能很强。输入图像已由转换器处理，此测试不会再去畸变，也不验收原始相机采集、CAN、硬件延迟、闭环任务成功率或未见数据泛化。比较多个 checkpoint 时保持数据、采样参数、随机种子及推理步数一致。当前 checkpoint 没有记录实际训练文件清单，因此 `--training-data` 的训练集身份由调用者保证。

## 操作顺序与按键

程序先检查标定、训练数据和相机配置，再加载模型、打开设备。模型预热一次，不执行这次预测。若配置了回 home，则完成后等待用户开始；预热和 home 期间的空格不会导致自动启动策略。

| 终端按键 | 作用 |
| --- | --- |
| 空格 | 开始 / 暂停 |
| `P` | 暂停 |
| `Q`、Esc、Ctrl+C | 退出 |

按键需要在**启动命令的终端**输入。预览窗口的 `Q` / Esc 只关闭预览，推理继续运行。

暂停会终止当前动作段并丢弃尚未执行的预测；恢复时重新获取实际图像和状态。正常执行完一个动作段后直接进入下一轮，不在每段边界重复回 home 或发送保持命令。

动作下发相对本段计划迟到超过 100 ms 时，程序先请求双臂保持；保持成功后进入暂停状态，丢弃该段剩余动作。终端按空格重新取图像与实际反馈并推理，按 `Q` 退出。程序不会补发旧段，也不会自动恢复。相机/机械臂反馈失效、SDK 下发失败、时钟异常或保持失败仍会停止程序。只观察模式不会发送保持或运动命令。

开启日志时，调度暂停在 `events.jsonl` 中记录为 `schedule_paused`，在 `cycles.jsonl` 中记录 `status=paused`、`pause_reason=schedule_slip`；内部 execution 保留超时原始错误及保持结果。`summary.json` 的 `schedule_pause_count` 汇总次数，关闭日志时也保留。每个动作的 `stage_timings` 记录键盘轮询、sleep 实际耗时及超时、验证和下发总耗时，`last_attempt` 保留未发送的失败动作；`commands.jsonl` 另记健康检查和 SDK 提交确认耗时。每个 chunk 保存 `measured_base_state`；每臂提交回执还保存当次确认循环读到的最新 `measured_feedback`，含末端位姿、关节位置、可用的关节速度、夹爪及来源时间戳。这不是发送瞬间的原子同步测量，反馈时间可能早于提交，也不表示已经到位。分析前先核对 `summary.json` 中 `logging.incomplete`；队列丢弃会造成动作索引缺口，不能再假定不同日志文件完整且逐行对应。

RTC 的 `cycles.jsonl` 同时保留原始 `camera_actions` / `base_actions`、段内滤波后的100点 `execution_base_actions`、`execution_transition_s` / `execution_smoothing_s` 和整窗 `transition_disagreement`。`execution_base_actions` 尚未包含运行时过渡采样；实际提交目标以 `commands.jsonl` 为准。`raw_switch_plan_delta` 衡量原始新预测与旧执行计划的同时间差，`switch_plan_delta` 衡量执行过渡后的同时间差，不能只看后者接近零便认定模型问题消失。拒绝接管和其他RTC暂停另见 `rtc_paused` 事件及摘要的 `rtc_pause_count` / `rtc_pause_reasons`。

这是同步实现：模型计算期间不会下发新预测。正在进行的一次模型计算不能被键盘中断；计算返回后再次检查按键，暂停或退出时丢弃结果。

## 训练与推理配置一致

四个路径均为必填参数：

| 参数 | 用途 |
| --- | --- |
| `--checkpoint` | 已保存的策略权重、归一化统计和训练动作时间配置 |
| `--calibration` | 转换训练数据使用的内外参 JSON，包含各臂的 `T_base_camera` |
| `--training-data` | 当前机器人 rig 的转换后 episode、批次目录或训练数据根 |
| `--camera-config` | 当前三路 UVC 采集设置及原始内参文件 |

预检只读 Zarr 元数据，不加载训练图像。它核对所有选中 PiPER episode 的外参、原始 K/D、图像处理、夹爪开度编码和 FPS，再检查当前相机角色、采集分辨率/帧率、曝光等控制值、延迟补偿和硬件身份。USB 端口可以变更，但必须更新相机配置中的当前设备路径和 `ID_PATH`；稳定硬件身份不能被删掉来绕过校验。当前相机存在重复序列号，front 与左腕的角色仍依赖正确的物理端口对应，软件不能靠序列号识别互换。

若训练数据目录包含多种机器人几何或图像预处理配置，程序会明确报错；将 `--training-data` 指向当前 rig 的子目录或单个 episode。混合目录里的非 PiPER 数据只统计、不当作机器人几何来源。**旧 checkpoint 未绑定实际标定 JSON 的 SHA**，所以预检以显式指定的训练数据为核对依据，不能仅凭权重证明选择的目录就是它的训练来源。

模型输入使用训练时保存的图像处理结果：当前 front 以保存的 K/D 和去畸变矩阵处理，`alpha=0`，输出 640×480；当前双腕只 resize，不会自动套用其它历史腕相机标定。原始输入与已处理图像明确区分，避免重复去畸变。

坐标变换始终由 JSON 决定，不硬编码单位外参：输入使用 `inverse(T_base_camera) @ T_base_end`，预测输出使用 `T_base_camera @ T_camera_end`。执行接口使用各臂 base 下的位置（米）、旋转向量（弧度）和夹爪宽度（米），调用 SDK 前再转换成连续欧拉角及 SDK 整数单位。单位矩阵和真实标定矩阵走同一条通路。

**当前 checkpoint 用单位外参训练，不能仅把推理参数换成真实外参来使用。** 后续完成真实外参标定、按正确坐标转换数据并重新训练后，使用同一入口传入相应标定和训练数据即可，适用于之后的 ego 混合训练。

## 动作频率与相机预览

第一版的循环是：获取实际观测 → 同步预测 → 按训练跨度重采样 → 执行动作前缀 → 获取新观测。

当前 checkpoint 的 30 个原始动作点在 30 Hz 下首尾相隔 `29 / 30 ≈ 0.967` 秒，训练插值成 100 点后跨度不变。推理从 checkpoint 的时间设置推导，恢复为 30 点，默认按目标 30 Hz 执行这 30 点；`--execute-steps` 控制的是**重采样之后**执行多少点，`--inference-steps` 是模型生成步骤数，二者互不替代。

每次模型计算后从动作块偏移 0 开始执行，不启用跳过过期预测点的推理延迟补偿。慢推理会延长两个动作段之间的间隔，实际频率不保证严格连续 30 Hz。相机配置里已有的图像延迟补偿仍照常用于观测对齐，与模型推理补偿是两件事。报告保存实际模型耗时和动作提交间隔，便于后续判断是否需要异步实现。

三路相机始终由独立进程采集。一个独立预览窗口从 SHM 读取左腕、front、右腕，默认每格 320×240、预览 15 fps；不另外打开相机。`--preview-fps 10` 可降低预览刷新率，`--no-preview` 可完全关闭，均不改变相机配置中的目标帧率。

第一版不要求配置工作空间和额外单步位移/旋转阈值；仍检查数值、维度、单位和来源更新时间。夹爪预测可能略超出物理行程，实时入口将两个夹爪命令截到 `[0, training_open_m]`（当前为 0–80 mm），不会修改机械臂位置或姿态。NaN/Inf 仍拒绝执行。原始预测保留在 `cycles.jsonl` 的 `camera_actions`；发生截断时，`events.jsonl` 的 `gripper_clamped` 记录原始宽度及数量，动作计划和提交记录保存处理后的值。

常用可选参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--device` | `cuda:0` | 模型设备 |
| `--control-frequency` | `30` | 动作重采样和下发目标频率 |
| `--execute-steps` | `30` | 每轮执行的动作前缀点数 |
| `--inference-steps` | `10` | 模型生成步骤数 |
| `--rtc-transition-ms` | `200` | RTC执行位姿过渡时长；0关闭过渡及分歧检查 |
| `--rtc-smoothing-ms` | `100` | RTC居中位姿滤波窗口约100ms；0关闭滤波 |
| `--rtc-max-position-gap-mm` | `20` | RTC完整过渡窗口允许的最大旧、新位置分歧 |
| `--rtc-max-rotation-gap-deg` | `5` | RTC完整过渡窗口允许的最大旧、新SO(3)角度分歧 |
| `--move-spd-rate` | `50` | PiPER SDK 的速度百分比，范围 1–100 |
| `--max-chunks` | 不限制 | 最多预测多少轮；不含启动预热 |
| `--max-observation-age` | `0.5` 秒 | 观测/相机/机器人来源最大允许年龄 |
| `--preview-fps` | `15` | 独立预览刷新率 |
| `--log` / `--no-log` | 开启 | 独立进程写运行日志；关闭时保留配置、结束摘要和终端提示 |
| `--output-dir` | 自动生成新目录 | 必须是尚不存在的目录 |

## 结果文件与验证范围

默认输出 `diagnostics/piper_live_<UTC时间>_<随机后缀>/`：

| 文件 | 内容 |
| --- | --- |
| `arguments.json` | 本次启动参数 |
| `manifest.json` | 权重和标定文件 SHA、训练集核对结果、K/D/T、相机配置、图像处理、动作时间设置 |
| `summary.json` | 完成/退出/错误状态、预测与动作数、模型耗时、实际提交间隔统计、日志完整性和丢弃计数 |
| `cycles.jsonl` | 开启日志时：每轮观测、预测耗时、重采样计划、执行数量及中断原因 |
| `events.jsonl` | 开启日志时：设备就绪、预热、home、开始/暂停/退出等事件 |
| `commands.jsonl` | 开启日志时：只观察的候选动作，或实际执行时两臂的 SDK 提交确认 |

SDK 提交确认表示主机侧两臂工作进程调用了 SDK 方法，**不等于机械臂已经运动到目标位置**。回执里的 `measured_feedback.ee_pose_xyz_rpy_m_rad` 是各臂base坐标下的SDK末端反馈，单位米/弧度，旋转为 `extrinsic_xyz_rpy`；关节速度不可用时记录为null。只观察日志明确标记没有发送机器人命令。加载或预检失败时仍保存参数和错误摘要，尚未完成的阶段不会产生完整 manifest。

当前新增执行连续性层尚未完成真实硬件运行验收。离线滤波/过渡对照、软件检查和只观察模式不能承诺真机平稳性或任务成功率，也不能修复模型错误目标；大分歧被拒绝必须单独报告。

不连接真机的连续性与RTC接入测试：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_policy_continuity.py tests/unit/test_policy_rtc_continuity.py \
  tests/unit/test_policy_rtc.py tests/unit/test_policy_rtc_timing.py \
  tests/unit/test_live_policy_devices.py
```

查看所有参数：

```bash
python -m egomimic.scripts.infer_piper_policy --help
```

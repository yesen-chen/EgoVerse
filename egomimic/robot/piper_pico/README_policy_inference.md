# Piper policy：同步推理与离线回放

现有两个入口，共用模型适配和坐标转换，但执行方式不同：

- `python -m egomimic.scripts.replay_piper_policy`：下文介绍的离线 CPU 回放，不打开硬件。
- `python -m egomimic.scripts.infer_piper_policy`：三 UVC、双臂反馈和终端键盘控制的实时同步推理，默认只观察，`--execute` 才允许运动。完整命令见 [实时推理说明](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_live_policy.md)。不需要 PICO 或 PCService。

本文命令、虚拟时间、工作空间参数和离线报告均针对 `replay_piper_policy`，不应直接套用于实时入口。实时第一版不要求工作空间/单步位移配置，也不启用下面描述的模型推理延迟补偿。

离线入口 `python -m egomimic.scripts.replay_piper_policy` 只读取本地转换后的
Piper episode 和可信的本地 checkpoint，在 CPU 上推理，把动作写入录制 sink。
不创建 `PiperPicoEnv`，不连接 CAN、相机或网络，也不调用回 home、上电或电机禁用。
这一步验证模型输入、坐标转换、动作时间与拒绝条件；不是策略真机成功率测试。

## 命令

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.replay_piper_policy \
  --checkpoint /absolute/path/to/trusted.ckpt \
  --episode /absolute/path/to/episode_000000.zarr \
  --output-dir /absolute/path/to/new_replay_report \
  --workspace-min "$LEFT_MIN_X" "$LEFT_MIN_Y" "$LEFT_MIN_Z" "$RIGHT_MIN_X" "$RIGHT_MIN_Y" "$RIGHT_MIN_Z" \
  --workspace-max "$LEFT_MAX_X" "$LEFT_MAX_Y" "$LEFT_MAX_Z" "$RIGHT_MAX_X" "$RIGHT_MAX_Y" "$RIGHT_MAX_Z" \
  --inference-steps 10 --control-frequency 30 --execute-steps 1 --max-chunks 1
```

上述 `LEFT_*`、`RIGHT_*` 必须由调用者明确设置，分别是两只机械臂自身 base
坐标系中的 xyz 米制范围。程序没有默认工作空间，也不会根据当前数据的
min/max 伪造硬件安全范围。checkpoint 必须可信：Lightning/PyTorch checkpoint
可能含 pickle 内容。只有显式 `--allow-synthetic-calibration` 才接受 synthetic
标定；这项开关不会把合成标定变成真实标定。

默认只推理一个 chunk，发送重采样后的第一个目标到内存录制 sink。增加
`--execute-steps` 才检查并回放更长的前缀；超过整个 chunk 的长度会截到末尾。
`--max-chunks` 控制最多取得多少个观测并预测。不给 `--output-dir` 时，程序在
`diagnostics/` 生成含 UTC 时间和随机后缀的新目录；显式目录已存在则拒绝。
每个结果文件用独占创建模式写入，不覆盖旧报告或修改源 episode/checkpoint。

输出包含：

- `report.json`：模型 timing、limits、标定状态、原始观测时间、每路来源时间的
  语义、CPU 实测推理耗时、控制状态变化、执行时间与拒绝原因。加载模型失败也
  写失败报告，进程返回非零。
- `scheduled_actions.jsonl`：重采样后准备检查的动作前缀，逐项标记是否送入 sink；
  已发送项附计划/实际发送时间和 slip。未发送目标不代表已经通过安全检查。
- `sent_actions.jsonl`：实际送入录制 sink 的 base-frame 动作与 freshness deadline。
  此处的 “sent” 仅指 Python sink 调用，不代表 CAN 发送、机器人执行或到位。

## 时间与数据含义

回放使用虚拟时钟：等待只推进离线调度时间，CPU 推理期间不推进虚拟时钟。
因此报告里的调度频率是离线名义时间验证，**不能证明 CPU 能满足实时推理**。
`measured_inference_elapsed_s` 是实际 CPU 墙钟耗时，独立记录；实时接入时必须
使用真实 wall/monotonic 时钟，推理后过期的观测会被拒绝。

原始 episode 时间戳保存在 `recording_observation.original_observation_timestamp`，
不会被伪称为当前时间。虚拟时间与原始时间间只有固定平移。v3 数据带有
`source.<camera>.timestamp` 和 `source.left|right.robot_timestamp`，保留各路来源
相对观测的真实偏差；缺失时明确标记为观测时刻 proxy。

图像解码为 RGB uint8 HWC；状态来自 `left/right.obs_ee_pose`，将 quat wxyz
转回 rotation vector，而不是读取 command 标签作为观测。夹爪宽度由
`obs_gripper * gripper_encoding.open_m` 恢复为米。推理适配器再用该 episode
的 `T_base_camera` 转换到训练坐标，最后将预测恢复成各臂 base 下的
`[xyz_m, rotvec_rad, width_m]`，顺序 left、right，共 14 维。

UVC 图像处理：转换器保存 `image_processing` 时，实时 `ObservationFrame`
默认 `image_space="raw"`，适配器使用保存的 K/D、K_new 去畸变和 resize。
`RecordedObservationProvider` 标记 `image_space="processed"`，已转换 JPEG
不会再次去畸变。图像尺寸与保存的处理记录、checkpoint 必须一致；原图与
处理后图像不可仅凭尺寸猜测。具体配置及现场验收见 `README_uvc_front.md`。

当前 v1 checkpoint contract 没有保存训练转换时的 `open_m`。报告中的
`gripper_contract.open_m` 来自本次 episode metadata，且
`checkpoint_open_m_verified=false`；硬件接入前须核对训练数据转换时使用
同一个开口宽度，不能仅凭模型 shape 或当前 metadata 推断一致。

ActionTiming 保持 checkpoint 中的动作跨度，xyz/夹爪线性插值，旋转用最短路径
SLERP；控制频率与 FM 推理步数独立。若跨度×控制频率不是整数，按最近整数
间隔数保留两端点，报告实际重采样频率。freshness deadline 只表示命令有效期，
不保证底层按时间执行。

延迟补偿：训练数据里，一行观测之后约 `time_alignment.observation_to_send_s`
（记为 Δ）发出该行命令。`PiperPolicy.observation_to_send_s` 从 episode metadata
读取 Δ，回放把它传给 `execute_action_block(observation_to_send_s=...)`。
执行时把发送墙钟 s 映射到动作块偏移 `s − τ − Δ`：推理比 Δ 快就等到 τ+Δ 再发
（`start_delay_s`）；比 Δ 慢就跳过已过期的部分，从对应偏移开始
（`start_offset_s`）；晚到超过整个动作块跨度则拒绝并 hold。每个发送样本记录
`observation_to_send_s`，可直接与训练分布对比。不传 Δ 时保持旧行为：推理完成
后从偏移 0 开始发送。

相邻 chunk 也保持最小命令间隔：下一次同步获取新观测并推理后，只等待距
上一次实际发送满一个实际重采样间隔所剩的时间，再次检查全部五路 freshness
后发送。因此一步前缀和快速推理不会造成跨块突发，慢推理已经超过此间隔时
不再额外等待整个周期。`inter_block_wait` 记录这段等待。等待中继续轮询
pause/stop；pause 会 hold 并丢弃已经计算的预测，恢复后重新观测和推理。
单个 chunk 的时间跨度保持不变。

## 同步接口与拒绝条件

`run_sync_cycle(policy, provider, sink, limits, ...)` 的 provider 实现 `latest()`，
每次返回 `ObservationFrame`；sink 实现 `send_action(row14, deadline_walltime)`
及 `hold(reason)`。`run_sync_policy(..., max_chunks=...)` 重复执行这些周期。
接口不负责构建设备；实时入口通过 `live_policy_devices.py` 连接真实 provider/sink。

真实 provider 必须提供同一时钟域下的 front_1、left_wrist、right_wrist、left、
right 五路独立来源时间戳；仅有对齐后的观测时间不足以识别某一路停更。
真机 provider 应直接用 `ObservationFrame.from_aligned_obs(env.get_obs(), camera_names=...)`：
`get_obs` 和采集时是同一个调用，返回选中帧和手臂最新样本的时间戳作为来源时间。
启动前用 `check_capture_config(metadata, env.camera_specs)` 确认每路相机的类型、
分辨率、帧率、eye 和延迟补偿与录制时一致；延迟不同会让选中帧整体错开。每次
推理前后均检查全部来源和观测时间；下一次预测必须取得更新的观测。两次预测
之间重用旧观测、非有限值、工作空间越界、夹爪越界或步长越界都会失败并 hold。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| workspace_min/max | 必填，两臂各 xyz | 绝对 base-frame 工作空间，米 |
| max_translation_step_m | 0.05 | 从测量状态到首目标、后续相邻命令的最大平移 |
| max_rotation_step_rad | 0.35 | 同上，SO(3) 最短旋转角 |
| gripper_min/max_m | 0 / 0.08 | 绝对夹爪宽度范围 |
| max_observation_age_s | 0.5 | 开始执行前观测/所有来源最大年龄 |
| future_timestamp_tolerance_s | 0.05 | 时钟/离线最近图像造成的未来时间容忍 |
| max_schedule_slip_s | 0.1 | 相对原计划累计延迟上限；超限拒绝追赶发送 |
| command_freshness_s | 0.1 | 每次单点发送的 freshness deadline 增量 |

这些是显式软件检查，不能代替碰撞检测、真实工作空间标定或固件保护。
容忍范围内的迟发也会等待下一完整间隔，不突发追赶。hold 在正常 chunk
结束、pause、stop、错误时都会调用；具体硬件 hold 行为由后续 sink 定义。

可选 `control_callback()` 返回 `run`/`None`、`pause` 或 `stop`，状态变化记录到
report。pause 立即结束当前 chunk 并 hold；重新 run 后重新取观测、重新推理，
不会续播旧 chunk。已耗尽 max_chunks 的暂停不会自动启动额外预测。stop 结束
整个循环。`replay_piper_policy` CLI 是离线批处理；CAN 和终端交互由独立的 `infer_piper_policy` 入口提供。

验证：

```bash
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_piper_policy_rollout.py tests/unit/test_policy_execution.py
```

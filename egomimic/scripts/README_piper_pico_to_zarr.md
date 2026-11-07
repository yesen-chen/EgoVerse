# Piper 真机数据 → Piper Zarr

入口：`python -m egomimic.scripts.piper_pico_to_zarr`。这是**真机**转换器；
不处理 PICO ego 人体数据，不连接硬件。读取一批采集目录中的
`replay_buffer.zarr`、`videos/cameras.json`、MP4 和逐帧 `.npy`。

输出的 `embodiment=piper_bimanual`（编号 7），schema 为 `piper_base_v3`。
输入必须是逐拍采集器录下的数据（replay buffer 里有 `camera<i>_timestamp`、
`decision_timestamp` 等逐拍字段）；更早的按命令标签插值的录制会被拒绝，
训练加载器也只接受 v3。
Human/Eva 原有编号 1–6 保持不变；低维和图像字段沿用原版 Eva 的布局，
机器人身份不再借用 Eva。`conversion_report.json` 同时记录身份、schema 和
`missing_required_cameras`。训练使用 Piper 专用加载器按 episode 读取标定。

旧测试产物的 `schema=piper_eva_base_v1`、`embodiment=eva_bimanual` 保持原样。
如需使用新身份，重新运行转换器并指定一个新的 `--output-dir`；不会原地修改旧数据。

## 使用

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.piper_pico_to_zarr \
  --input piper_pico_multicam_20261025_004052 \
  --output-dir converted_real \
  --calibration /path/to/this_batch_calibration.json \
  --task-name pick_place \
  --task-description 'Pick up the object and place it in the container.'
```

每次调用对应一个采集批次；也可在输入目录放 `calibration.json` 自动读取。
不同批次分别提供自己的标定。输出是
`converted_real/<输入目录名>/episode_000000.zarr`，默认转换所有已提交 episode；
`--episodes 0 2` 可选择。现有输出不会覆盖，失败不会留下冒充完整 episode 的 Zarr。

尚未标定时，可显式使用 `--allow-uncalibrated` 做格式和时间对齐测试。
此模式不填假矩阵，输出标记 `calibration_status=missing`，不代表已经可以正确训练。
老的一路/两路相机数据另加 `--allow-missing-cameras`；缺失视角不会复制其它图像填充。
未知相机名照原名保存，不能把名为 `test_uvc` 的流默认冒认为前视或某侧腕部相机。

## 外参文件

JSON 包含以下字段（矩阵必须替换为该批次的真实标定，下面为格式说明）：

```text
{
  "calibration_id": "采集安装配置的唯一标识",
  "reference_camera": "front_1",
  "extrinsics_convention": "T_base_camera",
  "extrinsics": {
    "left":  front_1相机在左臂基座中的4×4位姿,
    "right": front_1相机在右臂基座中的4×4位姿
  },
  "intrinsics": {
    "front_1": 原始front_1图像的3×3或3×4内参
  },
  "image_sizes": {"front_1": [1080, 810]}
}
```

`T_base_camera` 把相机坐标转换到相应基座；训练时应使用其逆变换 base → camera。
平移单位为米。两个外参验证为刚体变换，并原样写入每个 episode 的 `extrinsics`。
也可提供腕部内参（及对应原始图像尺寸）。默认输出 640×480 JPEG，与原版 Eva
一致；内参按缩放同步调整。不会通过分辨率猜内参，也不改变相机朝向/裁剪。
完整标定内容、来源路径和 SHA-256 保存在 metadata 的 `calibration` 中。

UVC front 的完整工作流见 `../robot/piper_pico/README_uvc_front.md`（仓库内文件为
`egomimic/robot/piper_pico/README_uvc_front.md`）。外参导出器还会写入原图
`distortion_coefficients`、`camera_models`、`raw_camera_calibration`。转换器自动
核对录制内参快照，去畸变后 resize；`--undistort-alpha` 默认 0，不裁 ROI。
根 metadata 的 intrinsics 对应最终 JPEG；原图 K/D 与 K_new 保存在
`image_processing`，以供推理使用同一处理。只有确实做了去畸变的视角输出 D=0。
缺少畸变标定的旧数据仍沿用 resize，不声称已校正畸变。

## 位姿与夹爪

转换**不应用外参、不使用原版 Eva 的旋转常量、不改变 TCP 原点**。
两臂位姿仍各自在自己的 Piper base 系下，旋转向量仅转换成 `quat(wxyz)`：

| 输出 | 来源 |
|---|---|
| `left/right.obs_ee_pose`，T×7 | `robot0/1_eef_pos` 与 `eef_rot_axis_angle` |
| `left/right.cmd_ee_pose`，T×7 | `action` 的左右目标位姿 |
| `left/right.obs_gripper`，T×1 | 本体开度 / 固定最大开度 |
| `left/right.cmd_gripper`，T×1 | `action` 中的目标开度 / 固定最大开度 |
| `left/right.*_gripper_width_m` | 原米制开度，供审计 |
| `left/right.obs_joints`（输入存在时） | 本体关节角，弧度 |
| `images.front_1/left_wrist/right_wrist` | 对应视频的选中帧，RGB JPEG |

`--gripper-max-m` 默认0.08；归一化0为闭合，1为全开。不会按每段最值缩放，
不会静默裁掉反馈超量程值；超范围次数写报告。没有伪造 `cmd_joints`。

原版 `Eva.get_transform_list()` 仍用类中硬编码外参。Piper 专用训练加载器从
**每个 episode**读取外参，校验图像处理记录；原版 `eva.yaml` 不能直接用于
Piper 数据。工具点仍是 Piper SDK 末端定义。

## 时间规则与诊断

一行就是采集时的一个控制拍，不重采样、不插值：

- `timestamp` 是该拍 `env.get_obs()` 的观测时刻 τ（读完设备缓冲区后的墙钟）。
  真机推理调用的是同一个 `get_obs`，所以训练和在线对「现在」的定义一致。
- 观测状态就是 `get_obs` 在 τ 取到的手臂状态（不晚于 τ 的样本）。
- 图像是 `get_obs` 选中的帧：每路相机不晚于 τ 的最新一帧（因果，绝不选更晚的帧）。
  转换器按记录的帧时间戳在 `.npy` 里精确匹配回视频帧号。录像开始前收到的帧
  只可能出现在开头几拍，这些拍会被裁掉（`trimmed_leading_rows`）；中间缺帧报错。
- `cmd_*` 是看到这一行观测之后才决策、立即下发的那条命令。动作块按行号取后续
  命令；行间名义间隔 1/fps，实际间隔统计在 `tick_interval`。
- 超过 `--max-tick-gap`（默认 50 ms）的间隔只记录在 `tick_gaps`，不拒绝、不切分。
- 选中帧比 τ 旧超过 `--max-frame-age`（默认 100 ms）、手臂状态比 τ 旧超过
  `--max-robot-age`（默认 50 ms）则报错，说明那一路设备停更。
- `time_alignment.observation_to_send_s` 是「观测到命令实际发出」的中位延迟，
  取自手臂进程回传的发送时刻；部署执行时按它对齐动作块的起点。
- 视频通过CPU顺序解码，校验每个视频的完整解码帧数与时间戳数量一致；仅保留选中帧
  的压缩JPEG，不把整段1080×810视频全部载入内存，不使用GPU。
- `.npy` 已经减过采集配置中的相机延迟，不重复扣减；部署时每路相机必须使用同样的延迟。
- `source.action_index/decision_timestamp/deadline_timestamp/command_id`、
  `source.left|right.robot_timestamp/send_timestamp`、
  `source.<camera>.frame_index/timestamp/age/receive_timestamp` 保留来源。
- `conversion_report.json` 和Zarr metadata记录拍间隔、帧龄、手臂状态年龄、观测到
  决策/发送延迟、未发出的命令数、重复帧等。ZarrWriter会补齐最后一个存储chunk，
  读取实际样本数应使用 `attrs.total_frames`。

真实标定参数尚未提供、缺失腕部视角、或采集时相机延迟为0的测试数据，转换成功
只能说明该段格式/读取/配帧通过，不能据此确认空间和曝光时间已经标定正确。

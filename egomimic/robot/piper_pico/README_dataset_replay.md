# 转换后 Piper 真机轨迹回放

入口：`python -m egomimic.scripts.replay_piper_dataset`。
它回放采集时的 command，不运行训练模型，与 `replay_piper_policy.py` 不同。

只接受 `piper_pico_to_zarr` 生成的 `piper_bimanual / piper_base_v3`。
Ego、Eva、相机系 pose、原始 `replay_buffer.zarr` 和其它格式都会报错。
所有选中的 episode 会在连接 CAN 前完整校验，不会跳过不兼容的数据。
不读取视频，不启动 PICO、ADB、相机或 PCService。

## 本次数据

先做无 CAN 验收（默认模式；虚拟时间，不用等待录像时长）：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.replay_piper_dataset \
  --input converted_real/piper_pico_multicam_20261025_004052 \
  --dry-run
```

真机回放：

```bash
python -m egomimic.scripts.replay_piper_dataset \
  --input converted_real/piper_pico_multicam_20261025_004052 \
  --left-can can0 \
  --right-can can1 \
  --approach-duration 4 \
  --speed 1 \
  --execute
```

`--input` 可以是一个 episode 的 `.zarr` 目录，也可以是一个转换批次目录。
批次内按路径排序逐条回放。也支持多个输入路径，按参数顺序处理；重复选择会报错。
指定单条示例：

```bash
python -m egomimic.scripts.replay_piper_dataset \
  --input converted_real/piper_pico_multicam_20261025_001025/episode_000001.zarr \
  --execute
```

## 每条轨迹的执行顺序

1. 读取当前双臂实测末端和夹爪。
2. 从当前位置平滑过渡到这一段的第一条 **command**，不是统一回 home。
   XYZ 和夹爪用平滑插值，旋转使用最短路径 SLERP；默认至少 4 秒。
   距离较远时会自动延长，使过渡目标的最大平移、转动、夹爪开度变化速度
   分别不超过 0.05 m/s、0.5 rad/s、0.04 m/s。
3. 等待两侧实测值连续 0.3 秒处于首点容差内：位置 10 mm、旋转 0.08 rad、
   夹爪开度 3 mm。最多等待 10 秒，未到位则停止，后续轨迹也不执行。
4. 首点到位之后重新开始计时，按该段 `source.decision_timestamp` 的相邻间隔
   下发全部 `cmd_ee_pose + cmd_gripper`。两臂使用同一回放时间轴。
5. 等待末点到位并保持；下一条重新执行以上过程。

如果夹爪被物体挡住、无法达到录制的首点或末点开度，也会触发到位超时。
`--position-tolerance`、`--rotation-tolerance`、`--gripper-tolerance`、
`--settle-duration`、`--settle-timeout` 可配置。

## 数据与时间含义

- `cmd_ee_pose` 是各自 Piper 基座系中的 `xyz(m) + quaternion(wxyz)`。
  发送前转成 SDK 的连续 XYZ Euler 分支；不会施加 Eva 的坐标变换。
- `cmd_gripper` 按本段 `gripper_encoding.open_m` 还原为米。超范围会拒绝，
  不会把归一化的 0–1 误当米。若保存了 `cmd_gripper_width_m`，两者必须一致。
- 因为回放直接使用基座系 command，`--allow-uncalibrated` 转换出的数据也能用，
  无需相机外参。不会把相机变换应用到回放轨迹上。
- 使用录制的决策间隔，保留短周期变化；不会拿相机 60 fps 当动作频率，
  也不使用已经过期的原始绝对时间／deadline。
- `--speed 0.5` 将所有回放间隔乘以 2，目标点数和路径不变。
  首点过渡速度独立控制。首版仅允许 `0 < speed <= 1`。
- SDK 的 `move_spd_rate` 默认从每段采集参数读取，缺失时为 50；
  `--move-spd-rate` 可覆盖。它与回放时间倍率 `--speed` 是两个不同参数。
- 发送按绝对单调时间调度，普通 sleep 误差不会逐点累加。轻微延误后，
  相邻发送至少相隔原间隔的一半；超过 `--max-schedule-slip`（默认 100 ms）
  则停止，不连续补发积压动作。SDK 调用本身不提供硬实时保证。
- Zarr 的末尾存储补齐行不回放，只读取 `total_frames` 条。
  回放的是记录的命令意图，不因末条缺少发送回执就擅自删除该 command。

## 中断、验收与边界

`Ctrl+C`、运行异常或反馈超过 250 ms 未更新时，尝试清除两臂待执行轨迹，
用新鲜实测值保持，然后释放 CAN；保持失败也会尝试释放两侧接口。
不会主动关闭关节力矩。过渡路径是末端空间插值，没有障碍物避让或逆运动学
可达性规划；真机使用时应确保当前位置到首点的路径畅通，且没有其它程序同时控制 CAN。

每次生成新的 `diagnostics/piper_dataset_replay_*.json`，也可通过 `--report` 指定。
包含每段首点过渡、实测到位、回放点数／时长、最大调度延误和失败原因。
默认无 CAN 验收使用第一段的首个记录状态作为模拟初始状态，随后模拟理想跟随；
它验证数据和执行流程，不证明真实机械臂的到位能力、运动路径或跟随误差。

测试：

```bash
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_piper_dataset_replay.py
```

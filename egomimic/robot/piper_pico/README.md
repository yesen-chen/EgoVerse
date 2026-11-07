# Piper + PICO 多相机采集

正式入口为 `egomimic/scripts/collect_piper_pico.py`。当前推荐 **LRCP UVC front + 双腕 UVC**，
PICO 继续提供手柄遥操。`calibrations/uvc_rig.json` 已更新为三路当前设备；
先通过相机录像检查，再启动 PCService 和 XRoboToolkit，显式使用统一配置：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.collect_piper_pico \
  --left-can can0 --right-can can1 --frequency 30 \
  --camera-config calibrations/uvc_rig.json \
  --output "piper_uvc_multicam_$(date +%Y%m%d_%H%M%S)"
```

默认有三相机小窗口（左腕 / front / 右腕，约 960×304），独立进程从 shm 读取最新图像，预览 15 fps。
加 `--no-preview` 关闭，或 `--preview-fps 10` 降低显示刷新率；这些选项不改变录像尺寸和相机/控制频率。
窗口 Q/Esc/关闭按钮只关闭预览；采集快捷键继续用 PICO 或终端（键盘需要终端焦点）。
无桌面自动跳过窗口。帧率含义和实现边界见 [预览说明](README_uvc_front.md#3-正式采集转换和推理图像一致性)。

当前 front 为 LRCP F1080P：640×480、30 fps、MJPG；三台 UVC 均正式采用亮度 48、手动曝光 20 ms，
rig 明确设置 `brightness=48`、`auto_exposure=1`、`exposure_time_absolute=200`，采集启动时应用并核对读回。
内参来自 `calibrations/uvc_front_lrcp_480p_002_result/calibration.json`，延迟补偿约 67.3 ms。
后者来自此前亮度 32、自动曝光模式 3 的单相机录像测试，含未独立测量的屏幕延迟。两腕延迟补偿保留 66 ms；
右腕来自此前测量，左腕沿用右腕值、尚未独立测量。
单 front 已可使用 `calibrations/uvc_front_lrcp_480p_002_camera.json` 独立测试及标定。
USB 接口已确认：front `5.3.2`、左腕 `5.3.3`、右腕 `5.3.4`，均使用 `video-index0`。
三台序列号重复，front 与左腕的 USB 标识也相同；继续按接口绑定，换线后预览核对视角。
此前自动曝光检查中，front 三路录像约 22.06 fps，单独也只有约 20.24 fps；曝光对照确认固定 20 ms 后恢复约 30 fps。
三台统一 20 ms、亮度 48 的 15 秒录像现已通过：front/左腕/右腕分别为 30.018/30.015/30.013 fps。
报告为 `diagnostics/uvc_all_20ms_brightness48_preview_001/capture/report.json`，该预览设置已正式写入配置。
本次按用户决定沿用上述延迟补偿，不重测；配置记录新曝光下延迟未测，不将历史补偿描述为新设置的实测值。
旧 KD front 已损坏，其几何标定不能用于 LRCP。当前相机需要自己的左右 base 外参。
完整配置、检查和转换流程见 [UVC 三相机说明](README_uvc_front.md)。省略相机参数不会自动加载 rig。

## PICO 图像采集（历史/可选模式）

以下保留 PICO 图像的独立用法，不是当前默认 front 配置。`--headset-ip` 启用
PICO **左目**，默认名称 `front_1`、请求 `1080×810@60`。每个 `--camera`
添加一只 UVC，相机名称和设备不能重复。控制频率与相机帧率分别设置。

PICO 左目加一只 UVC（真机遥操作，需先运行 PCService 和 XRoboToolkit）：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
./emimic/bin/python egomimic/scripts/collect_piper_pico.py \
  --left-can can0 \
  --right-can can1 \
  --headset-ip 192.168.1.9 \
  --pc-ip 192.168.1.8 \
  --pico-camera-name front_1 \
  --pico-camera-fps 60 \
  --camera left_wrist=/dev/video0 \
  --camera-fps 30 \
  --width 640 \
  --height 480 \
  --fourcc MJPG \
  --frequency 30 \
  --pico-camera-latency 0.1 \
  --camera-latency 0.08 \
  --output "piper_pico_multicam_$(date +%Y%m%d_%H%M%S)"
```

增加第二只 UVC 时，再加一项 `--camera right_wrist=/dev/video2`。
`/dev/video2` 只是示例：先用 `v4l2-ctl --list-devices` 确认第二只相机的
图像采集节点，不要选第一只相机的 metadata 节点。也支持 `/dev/v4l/by-path/...-video-index0`。
当前三台 LRCP 的 `by-id` 名称存在冲突，不能用它区分左右腕和 front。
未传 `--headset-ip` 时仍支持原来的纯 UVC 图像采集，PICO 手柄遥操作照常使用。

- B：三路一起开始录制；再次按 B：关闭各路视频并保存同一段低维数据。
- X：停止并丢弃当前段的所有视频和低维缓存。
- A：保存正在录制的段，然后退出并关闭头显相机。
- Y：双臂回 home；Grip：离合。

PICO 图像通过现有 TCP 收流程序接收，使用的端口仍为头显控制 `13579`、
PC 视频接收 `12345`。保持 XRoboToolkit 运行，关闭头显本地 Record 和其它
相机预览。这里只启用左目输出，没有申请右目视频。

每路独立采集。PICO 每个解码帧写入共享内存，同时送到其相机进程内的
异步录像线程；不会只保存 30 Hz 控制循环看到的帧。队列最多缓存 32 帧，
编码或磁盘跟不上时会报错并拒绝提交不完整的数据段，不会静默丢帧。
实际三路持续帧率仍需联机实测，命令中的 fps 是请求值。

文件布局（有 PICO 时排在第一路，UVC 按命令参数顺序）：

```text
output/
  replay_buffer.zarr
  videos/
    cameras.json              # ["front_1", "left_wrist", "right_wrist"]
    camera_config.json        # 来源、分辨率、请求帧率、延迟补偿等
    0/
      0.mp4                   # PICO 左目
      0.npy                   # 与视频逐帧对应的校正时间戳
      0.receive_timestamps.npy # PICO 原始主机接收/解码时间戳
      1.mp4                   # 左腕 UVC
      1.npy
      2.mp4                   # 右腕 UVC
      2.npy
      robot0_state.npz        # 左臂完整状态流（含命令编号和实际发送时刻）
      robot1_state.npz
```

各路视频起止时刻不保证完全一致，也不代表硬件曝光同步。MP4 播放帧率来自
请求值，实际采集帧率须用 `.npy` 测量。

`replay_buffer.zarr` 每行是一个控制拍：先调用和真机推理相同的 `env.get_obs()`，
记下观测时刻 τ、每路相机选中的帧（不晚于 τ 的最新帧）的时间戳和该时刻的
手臂状态；再计算并立即下发命令，记下决策时刻和命令编号。手臂进程回传每个
命令的实际发送时刻。转换器据此原样输出逐拍训练行，不再重采样。超时的一拍
不会补发，下一拍从当前时刻重新计时。

`--pico-camera-latency` 单独指定左目延迟，`--camera-latency` 指定 UVC
延迟（旧参数方式对全部 UVC 应用同一值），单位秒。上述历史示例使用
PICO 左目 100 ms、UVC 80 ms：`.npy = 主机接收/解码时间 - 指定延迟`。
历史示例的参数不代表当前三台相机的实测值，旧的 77 ms / 130 ms 也不适用。
当前 20 ms 配置按用户决定沿用 front 67.275 ms、两腕 66 ms，记录其历史来源及新设置未复测的状态。需要各 UVC 独立补偿时，
使用当前推荐的 `--camera-config`，在 JSON 中分别设置每路 `latency`。
单路测量方法和命令见 [相机延迟标定](README_camera_latency.md)；显示器刷新率
只作为记录信息，实际显示延迟必须独立测量后显式传入，不能直接扣一个刷新周期。

UVC 现在显式使用 V4L2，并检查分辨率、格式和帧率的模式读回值。
这不保证自动曝光下的实际输出帧率。此前测试的 LRCP F1080P 在当前场景
自动曝光下约 20 fps，手动曝光 20 ms 可达 30 fps。当前正式 rig 在采集启动时
为三台应用曝光及亮度控制，不再依赖硬件保留状态；其它 UVC 的控制名称和值应另行查询。

已有 output 的相机顺序或配置改变时会拒绝继续追加，以免改变旧视频编号的
含义；新增相机请使用新的 output 目录。录像打开/收尾失败时，不提交这段
到 zarr，保留部分视频供检查；不要把这些未提交文件当作完整 episode。

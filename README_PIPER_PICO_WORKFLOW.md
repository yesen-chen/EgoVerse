# PICO + PiPER 操作命令总表

推理诊断记录：在现有推理命令末尾加 `--log --record-images all --record-feedback`，保存三路连续画面、无损请求快照、动作及连续 TCP 反馈。配置、目录与无硬件压力测试见 [推理记录说明](egomimic/robot/piper_pico/README_policy_recording.md)。

本机工作流：**启动电脑服务 → 测试头显/相机 → 双臂外参标定 → 真机采集 → 转换数据 → 检查训练输入 → 训练**。
另收录 PICO 人手数据导出、历史投影复现、离线模型推理、实时模型推理和录制轨迹回放。前置相机工作流按 2026-11-05 的替换配置更新。

当前默认前置相机是 **LRCP F1080P，640×480@30 MJPG**，三台 UVC 正式统一为 **亮度 48、手动曝光 20 ms**。
旧 KD front 已损坏。单 front 使用 `calibrations/uvc_front_lrcp_480p_002_camera.json`，PICO 仍负责手柄遥操。
front 内参为 `calibrations/uvc_front_lrcp_480p_002_result/calibration.json`，延迟补偿约 67.3 ms；
新相机须重新求左右 base 外参，不能套用旧 KD 或 PICO 的结果。
三相机 `calibrations/uvc_rig.json` 已更新，接口经现场确认：**front `5.3.2`、左腕 `5.3.3`、右腕 `5.3.4`**。
每路配置均明确设置 `brightness=48`、`auto_exposure=1`、`exposure_time_absolute=200`，启动采集时应用并核对读回，不再依赖设备保留的曝光状态。
此前 front 自动曝光时三路录像约 22.06 fps、单独约 20.24 fps；曝光对照确认固定 20 ms 后恢复约 30 fps。
三台统一 20 ms、亮度 48 的 15 秒录像已通过，front/左腕/右腕分别为 30.018/30.015/30.013 fps。
报告见 `diagnostics/uvc_all_20ms_brightness48_preview_001/capture/report.json`，这组预览设置现已作为正式配置。
按当前决定不重新测延迟：继续使用 front 67.275 ms、两腕 66 ms；配置注明沿用历史结果，新曝光下尚未测量。
front 结果来自此前亮度 32、自动曝光模式 3 的测试；左腕的数值沿用右腕，尚未独立测量。
详细配置和硬件标识限制见 [UVC 三相机说明](egomimic/robot/piper_pico/README_uvc_front.md)。

这份文档是操作手册，按需执行各节，不要整份粘贴运行。标为“真机”的采集、标定、实时推理和回放命令会连接机械臂；其余离线步骤不需要机器人开机。

## 目录

- [0. 环境、地址和输出路径](#0-环境地址和输出路径)
- [1. 启动电脑服务与头显应用](#1-启动电脑服务与头显应用)
- [2. CAN 与腕部相机准备](#2-can-与腕部相机准备)
- [3. 头显相机、遥操作与休眠测试](#3-头显相机遥操作与休眠测试)
- [4. 测量相机延迟](#4-测量相机延迟)
- [5. front 相机到双臂 base 的外参标定](#5-front-相机到双臂-base-的外参标定)
- [6. 合并左右臂标定结果](#6-合并左右臂标定结果)
- [7. 采集真机数据](#7-采集真机数据)
- [8. 转换真机数据](#8-转换真机数据)
- [9. 导出和转换 PICO 人手数据](#9-导出和转换-pico-人手数据)
- [10. 可视化与训练输入检查](#10-可视化与训练输入检查)
- [11. 模型训练与恢复](#11-模型训练与恢复)
- [12. 模型推理与真机轨迹回放](#12-模型推理与真机轨迹回放)
- [13. 历史命令与详细说明](#13-历史命令与详细说明)

## 0. 环境、地址和输出路径

每个操作终端先执行下面的环境设置。IP、CAN 对应关系、相机节点按当天实际情况修改；下列 `.9/.8` 是之前用过的地址，不能保证今天仍相同。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source /home/yesen/pico_ego_probe/EgoVerse/emimic/bin/activate

PICO_WORKSPACE=/home/yesen/pico_ego_probe
EGOVERSE_REPO="$PICO_WORKSPACE/EgoVerse"
PICO_IP=192.168.1.9
PC_IP=192.168.1.8
LEFT_CAN=can0
RIGHT_CAN=can1
CAMERA_RIG="$EGOVERSE_REPO/calibrations/uvc_rig.json"
FRONT_CAMERA_CONFIG="$EGOVERSE_REPO/calibrations/uvc_front_lrcp_480p_002_camera.json"
# 已确认的腕部接口；保持插口不变即可避免 /dev/videoN 编号变化。
LEFT_WRIST=/dev/v4l/by-path/pci-0000:0f:00.0-usb-0:5.3.3:1.0-video-index0
RIGHT_WRIST=/dev/v4l/by-path/pci-0000:0f:00.0-usb-0:5.3.4:1.0-video-index0

# 每次重新安装相机/底座，换一个标定名称；每批新录制，换一个 RUN_NAME。
CALIB_NAME=lrcp_front_480p_rig_001
RUN_NAME=pick_place_001
LEFT_SESSION="$EGOVERSE_REPO/calibrations/${CALIB_NAME}_left"
RIGHT_SESSION="$EGOVERSE_REPO/calibrations/${CALIB_NAME}_right"
LEFT_RESULT="$LEFT_SESSION/calibration_result.json"
RIGHT_RESULT="$RIGHT_SESSION/calibration_result.json"
CALIBRATION_JSON="$EGOVERSE_REPO/calibrations/${CALIB_NAME}_combined.json"
ROBOT_RAW="$EGOVERSE_REPO/recordings/$RUN_NAME"
ROBOT_ZARR_ROOT="$EGOVERSE_REPO/data/piper_converted"
EGO_ZARR_ROOT="$EGOVERSE_REPO/data/pico_ego_converted"
```

这些变量只在当前终端生效。重开终端时使用**相同的实际名称**重新设置；转换时 `ROBOT_RAW` 必须指向已经完成的录制目录，不能换成另一个未采集的新名字。命令中的 `/绝对路径/...` 需要替换后使用。

查看设备与网络地址：

```bash
adb devices -l
ip -4 addr
adb shell ip -4 addr show wlan0
```

## 1. 启动电脑服务与头显应用

电脑上的服务实际安装在 `/opt/apps/roboticsservice/`。先检查，已经运行则不要重复启动：

```bash
pgrep -af RoboticsServiceProcess
```

未运行时启动，再检查监听端口：

```bash
bash /opt/apps/roboticsservice/runService.sh
ss -lntp | rg ':(63901|60061)\b'
```

用 `bash` 执行，不要 `source`。脚本会设置自己的库路径并后台启动服务，不需要一直占用一个前台终端。

启动头显 XRoboToolkit：

```bash
adb shell monkey -p com.xrobotoolkit.client \
  -c android.intent.category.LAUNCHER 1
adb shell pidof com.xrobotoolkit.client
```

在头显中连接 `PC_IP`，确认 `WORKING`、`Controller ON`、`Send ON`；使用头部跟踪时也打开 `Head ON`。若有 `Switch w/ A Button`，关闭它，避免头显应用截获脚本要使用的 A 键。

| 链路 | 端口 |
| --- | --- |
| 头显 → 电脑 PCService | TCP 63901 |
| 本机 Python SDK → PCService | TCP 60061 |
| 头显相机控制 | TCP 13579 |
| 电脑接收相机视频 | TCP 12345 |

控制器持续全零时，先停止采集，再重启头显应用并重新确认连接和 Send：

```bash
adb shell am force-stop com.xrobotoolkit.client
adb shell monkey -p com.xrobotoolkit.client \
  -c android.intent.category.LAUNCHER 1
```

## 2. CAN 与腕部相机准备

### 2.1 CAN

先确认实际接口名称及左右臂对应关系。不要把 USB 插入顺序当作永久的左右臂标识。

```bash
ip -br link show type can
ip -details link show "$LEFT_CAN"
ip -details link show "$RIGHT_CAN"
```

本机已有初始化脚本。仅在接口确为 `can0/can1`、且没有采集或控制程序运行时执行；它会关闭再启用接口，设置为 1 Mbps，必要时请求 sudo：

```bash
bash /home/yesen/cys/avfrl_real/setup_can.sh 0 1
```

### 2.2 UVC 图像节点与曝光

```bash
v4l2-ctl --list-devices
v4l2-ctl -d "$LEFT_WRIST" --list-formats-ext
v4l2-ctl -d "$LEFT_WRIST" --list-ctrls
v4l2-ctl -d "$RIGHT_WRIST" --list-formats-ext
v4l2-ctl -d "$RIGHT_WRIST" --list-ctrls
```

选择 `video-index0` 图像采集节点，不要选 metadata 节点。当前三台 LRCP 的序列号相同，
已经出现 `by-id` 名称冲突，不能靠该名称区分三台相机；使用上面已确认的 `by-path` 接口。
换 USB 口时同步更新 rig 中的 `device` 和 `expected_identity.ID_PATH`，并通过预览核对视角。

正式 rig 已为三台设置手动曝光 20 ms、亮度 48，采集启动时会应用并核对，无需另行手动设置。下面保留单独检查或设置的方式；先退出占用该相机的程序：

```bash
v4l2-ctl -d "$LEFT_WRIST" --set-ctrl=auto_exposure=1
v4l2-ctl -d "$LEFT_WRIST" --set-ctrl=exposure_time_absolute=200,brightness=48
v4l2-ctl -d "$LEFT_WRIST" --get-ctrl=auto_exposure,exposure_time_absolute,brightness
```

右腕可将设备变量改为 `RIGHT_WRIST`。`exposure_time_absolute=200` 对应 20 ms；其它设备使用前先查询控制项。此前 front 自动曝光下实际只有约 20 fps，手动 20 ms 时能达到 30 fps；最终仍要看采集时间戳，命令中的 fps 是请求值。

## 3. 头显相机、遥操作与休眠测试

本节保留 PICO 图像测试用法；当前 UVC front 的检查使用 `check_camera_rig --camera-config "$FRONT_CAMERA_CONFIG"`，
完整命令见 [UVC 三相机说明](egomimic/robot/piper_pico/README_uvc_front.md)。无需为了 UVC 图像启动 PICO 收流。

PC 相机收流期间，关闭头显本地 Record，并退出其它相机预览、延迟测试或采集程序。下面的相机命令依次运行，不要同时占用视频端口。

### 3.1 独立 RGB 预览：之前测试相机的工具

这套工具使用自己的环境，不需要 PCService 的姿态接口；头显 XRoboToolkit 相机服务仍需运行。

```bash
/home/yesen/pico_ego_probe/xrobo_pc_viewer_v1_20261009/preview.sh --check

/home/yesen/pico_ego_probe/xrobo_pc_viewer_v1_20261009/preview.sh \
  --headset-ip "$PICO_IP" --pc-ip "$PC_IP" \
  --width 2160 --height 810 --fps 60 \
  --view stereo
```

按 `1` 显示双目、`2` 左目、`3` 右目、`S` 保存原始截图、`Q/Esc` 退出。这里 `--view left` 只是显示双目图的左半张，**不是头显真正单目输出**。正式采集与标定使用真正单左目 `1080×810`。

无窗口收流 10 秒并保存首帧：

```bash
/home/yesen/pico_ego_probe/xrobo_pc_viewer_v1_20261009/preview.sh \
  --headset-ip "$PICO_IP" --pc-ip "$PC_IP" \
  --width 2160 --height 810 --fps 60 \
  --seconds 10 --headless --snapshot-first
```

结果在 `xrobo_pc_viewer_v1_20261009/runs/preview_*/`。可加 `--save-h264` 保留原始码流。离线播放之前的本地录制：

```bash
/home/yesen/pico_ego_probe/xrobo_pc_viewer_v1_20261009/preview.sh \
  --replay /home/yesen/pico_ego_probe/xrobo_record_20261029_213853_Rjdni9/CameraRecord_20261029_213853.mp4 \
  --view left --start-sec 5
```

### 3.2 手柄、虚拟双臂与单左目收流

这组命令需要 PCService 和头显控制器连接，但不打开 CAN、不运动真机。

```bash
# 仅手柄与虚拟机械臂
python egomimic/scripts/sim_piper_pico.py --frequency 30
```

加入真正的单左目收流与分段耗时记录：

```bash
python egomimic/scripts/sim_piper_pico.py \
  --headset-ip "$PICO_IP" --pc-ip "$PC_IP" \
  --camera-mode left --camera-fps 60 --frequency 30 \
  --profile-loop
```

窗口显示机械臂 3D，不是 RGB 预览；图像用于收流和帧率统计。加 `--no-view` 关闭 3D 窗口。Grip 离合、Y 双臂 home、A 退出。`--frequency` 可选 `30/60/90`，与相机 fps 独立；输出当前目录的 `piper_pico_sim_*.csv` 和 `*_timing.json`。

### 3.3 防休眠与检查

需要持续发唤醒键时，在单独终端运行，Ctrl+C 停止：

```bash
bash /home/yesen/pico_keep_awake.sh
```

只读检查 90 秒：

```bash
python egomimic/scripts/check_pico_awake.py --seconds 90
```

多台 ADB 设备时，检查命令加 `--serial 实际序列号`。报告在 `diagnostics/pico_awake_check_*/summary.json`。防休眠脚本不是 tracking 连续性的保证；检查 PASS 只表示观察期间没有睡眠。

## 4. 测量相机延迟

本次正式采用 20 ms、亮度 48 时选择沿用历史补偿，不重复测量；下列命令保留供之后按需复核，不是当前采集的前置步骤。

这是时间延迟测量，与第 5 节的几何外参标定分开。将相机对准电脑屏幕中的完整二维码，保持清晰约 10 秒，在 OpenCV 窗口按 `c` 结束采集并分析；`q/Esc` 退出。

以下 `120` 应改为显示器实际刷新率；该参数只记录信息，不修改屏幕、不自动扣除一个刷新周期。没有独立显示延迟测量时，保持 `--display-latency-ms 0`。

```bash
# 当前 LRCP front；本配置只含 front，不包含两腕
python -m egomimic.scripts.calibrate_camera_latency \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  uvc --camera-config "$FRONT_CAMERA_CONFIG" --camera-name front_1
```

```bash
# PICO 单左目：历史/可选图像模式，不是当前 UVC front
python egomimic/scripts/calibrate_camera_latency.py \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  pico --headset-ip "$PICO_IP" --pc-ip "$PC_IP" \
  --mode left --width 1080 --height 810 --fps 60 --bitrate 5000000
```

```bash
# 左腕 UVC，沿用正式采集的相机设置；右腕改为 right_wrist
python egomimic/scripts/calibrate_camera_latency.py \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  uvc --camera-config "$CAMERA_RIG" --camera-name left_wrist
```

全局选项必须放在 `pico/uvc` 子命令前。结果自动存入新的 `camera_latency_pico_*` / `camera_latency_uvc_*` 目录，包含 `latency_report.json`、`samples.csv`、`latency.png` 和录像。

front 历史三次测量（亮度 32、自动曝光模式 3）共 226 个有效二维码，合并均值 `0.067275 s`，中位数 67.107 ms、P95 85.188 ms。
这仍含未测的显示延迟。每路补偿由 rig 的 `latency` 分别指定，两腕不套用 front 的结果。
当前继续使用这些历史补偿，并记录新曝光下延迟未测；这不表示已验证曝光变化对延迟没有影响。测量程序不会自动改配置。`--record-video` 是单路录像负载，
不代表三路同时录制的延迟。旧 PICO 示例的 `0.1 s`、旧 UVC 示例的 `0.08 s` 不适用于当前 front。

## 5. front 相机到双臂 base 的外参标定

### 5.1 当前约定

- 眼在手外：标定、采集、推理期间，front 相机与双臂底座相对固定。
- 两个会话都使用当前单 front 配置的 **LRCP `front_1`**；每次 `--arm` 只选一只手臂采样，双臂遥操保持原样。
- 实物板是 **6×6 格，19 mm/格，网格 114×114 mm**；标记边长按同比例缩放为 **13.3 mm**，字典 `DICT_5X5_100`，`legacyPattern=False`。不要用 PDF 标称的 20 mm。
- 一个会话内板与末端固定，不能松动、重新夹持。换到另一只手臂建立新会话。
- 使用 SDK 实际返回的末端坐标系；没有增加指尖 TCP 偏移。

LRCP 的 K/D 自动从 rig 的 `calibration_file` 读取。使用原始 640×480 图像，不要把去畸变后的
`rectified_camera.json` 当作原图内参。保持本次标定的镜头设置不变。

<details>
<summary>PICO 图像历史模式：原左目内参和采样参数</summary>

只有明确改用 PICO 图像时才使用下面的参数，不能用于 LRCP。PICO 单左目 `1080×810` 的 K 横向参数不再除以 2：

```text
K = [[686.8340695362958, 0, 539.75],
     [0, 686.8680648828482, 404.5],
     [0, 0, 1]]
D = [0, 0, 0, 0, 0]
```

零畸变是该 PICO 输出图像模式的使用假设。PICO 采样时将下面的 `--camera-rig ... --camera-name front_1`
替换为 `--headset-ip "$PICO_IP" --pc-ip "$PC_IP" --pico-camera-fps 60 --pico-camera-latency 0.1`，
并设置独立的 PICO 会话目录。两种来源不能同时传。PICO 自定义内参可用 `--camera-config` 传入；
UVC 原始内参应放在 rig 中，不能使用这个旧 PICO 参数。

</details>

### 5.2 采集左右臂样本（真机）

先完成服务、CAN、头显连接。关闭其它采集程序；以下两条命令依次运行，期间将板牢固装到对应手臂。

```bash
python egomimic/scripts/calibrate_piper_pico.py collect \
  --arm left \
  --camera-rig "$FRONT_CAMERA_CONFIG" --camera-name front_1 \
  --left-can "$LEFT_CAN" --right-can "$RIGHT_CAN" \
  --frequency 30 --output "$LEFT_SESSION"
```

```bash
python egomimic/scripts/calibrate_piper_pico.py collect \
  --arm right \
  --camera-rig "$FRONT_CAMERA_CONFIG" --camera-name front_1 \
  --left-can "$LEFT_CAN" --right-can "$RIGHT_CAN" \
  --frequency 30 --output "$RIGHT_SESSION"
```

会话目录必须是新的；省略 `--output` 可自动命名。默认显示检测预览，无图形界面可加 `--no-preview`。

| 输入 | 标定脚本 | 第 7 节正常采集脚本 |
| --- | --- | --- |
| X | 左臂 home | 丢弃当前录制段 |
| Y | 记录左臂样本，仅 left 会话接受 | 双臂 home |
| A | 右臂 home | 保存当前段并退出 |
| B | 记录右臂样本，仅 right 会话接受 | 开始/停止并保存一段 |
| Grip / Trigger | 原遥操离合 / 夹爪 | 原遥操离合 / 夹爪 |

标定时电脑键盘 `V` 切换下一条为 TRAIN/VALIDATION，`U` 作废最近有效样本但保留原文件，`Q/Esc/Ctrl+C` 退出。**松开 Grip 不会冻结 Trigger 的夹爪控制**，夹板时保持所需 Trigger 输入。home 默认暂停双臂遥操更新 3 秒，目标臂延续此前有效夹爪命令。

每臂建议 30–40 个 TRAIN 和 8–10 个独立 VALIDATION 姿态，覆盖不同位置、距离和多轴倾角。按记录键后等待稳定，程序固定图像和当前 SDK 实测 pose，再后台检测、保存；不调用 `get_obs()`、不插值 pose。默认至少 16 个角点、单帧 PnP RMS ≤0.8 px，预览显示拒绝原因。这个门限不等于最终外参精度。

### 5.3 求解并自动输出 SDK 末端三轴图片（离线）

```bash
python egomimic/scripts/calibrate_piper_pico.py solve --session "$LEFT_SESSION"
python egomimic/scripts/calibrate_piper_pico.py solve --session "$RIGHT_SESSION"
```

分别得到 `calibration_result.json`。`T_base_camera` 将相机坐标变到该臂 base，平移单位米；同时保存逆矩阵和 `T_end_board`。只拟合 TRAIN，VALIDATION 用于留出检验。

**每次成功求解自动建立新的图片目录：**

```text
某一臂会话/
  session.json
  samples/000001/{raw.png,overlay.jpg,sample.json,feedback.npz}
  calibration_result.json
  verification/时间戳_唯一编号/
    validation/
      000033_sdk_end.png       # 相机原图 + SDK 末端三轴
      000033_board_check.png   # 机器人预测板角点与图像检测角点的偏差
      contact_001.jpg          # 缩略总览
    train/                    # 同样两类图片
    summary.json               # 完整链路像素误差及未生成图片的原因
    calibration_used.json      # 本批图片使用的标定结果副本
    README.txt
```

轴默认长 40 mm，X 红、Y 绿、Z 蓝。先看 `validation/`：板角点预测来自机器人 pose 和标定结果，而非直接把单帧 PnP 画回原图。图像叠加帮助目测，仍需结合留出误差及独立物理测量判断所需精度。

自定义图片目录/轴长时，例如：

```bash
python egomimic/scripts/calibrate_piper_pico.py solve \
  --session "$LEFT_SESSION" \
  --visualization-dir "$LEFT_SESSION/verification/custom_50mm_001" \
  --axis-length-mm 50
```

自定义目录必须不存在。重新 solve 会更新结果 JSON，但此前的图片及其结果副本保留。相机或底座移动后重标；单独重新夹板不改变已完成的相机到 base 外参，但新旧夹持样本不能混合求解。

### 5.4 单张标定板图像检测

```bash
python egomimic/scripts/calibrate_piper_pico.py check-image \
  --camera-rig "$FRONT_CAMERA_CONFIG" --camera-name front_1 \
  --image /绝对路径/front_raw_640x480.png \
  --output /绝对路径/新的单图检测目录
```

输出 `detection.json` 和 `overlay.jpg`。输入要是与 K 对应的相机原图。手拿板、没有机器人反馈的录像只能检查图像检测/板到相机位姿，不能求机器人 base 外参。

## 6. 合并左右臂标定结果

转换器 `--calibration` 需要同时包含左右臂外参及 front 内参，**不能直接传单臂 `calibration_result.json`**。
当前 LRCP 使用下面的导出入口；两臂结果必须是这台 LRCP 在同一固定安装下重新采集、求解并验证的结果。

```bash
python -m egomimic.scripts.calibrate_piper_pico export \
  --left-result "$LEFT_RESULT" --right-result "$RIGHT_RESULT" \
  --camera-rig "$FRONT_CAMERA_CONFIG" --camera-name front_1 \
  --calibration-id "$CALIB_NAME" --output "$CALIBRATION_JSON"
```

<details>
<summary>历史 PICO 左目模式：手动合并片段（不用于 LRCP）</summary>

仅用于原 PICO 零畸变图像流程。下面的限制是历史片段的校验要求，当前 UVC 转换器已有畸变处理。

它保留原结果、路径和哈希，检查单位、矩阵及相机参数，不代替物理验收；不会自动把结果标记为 production approved。目标 JSON 已存在时拒绝覆盖。

```bash
python - "$LEFT_RESULT" "$RIGHT_RESULT" "$CALIBRATION_JSON" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

paths = {"left": Path(sys.argv[1]).resolve(), "right": Path(sys.argv[2]).resolve()}
destination = Path(sys.argv[3]).resolve()
results = {side: json.loads(path.read_text()) for side, path in paths.items()}
for side, result in results.items():
    if result["arm"] != side or result["camera_eye"] != "left":
        raise ValueError(f"{side}: 手臂或相机身份不匹配")
    if result["translation_unit"] != "metre":
        raise ValueError(f"{side}: 平移单位必须为米")
    camera = result["camera"]
    if camera.get("eye", "left") != "left":
        raise ValueError(f"{side}: 必须使用 PICO 左目")
    K = np.asarray(camera["K"], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError(f"{side}: 内参格式不正确")
    if not np.allclose(camera.get("D", [0.0] * 5), 0):
        raise ValueError("当前转换/训练链路不校正非零畸变，需先统一图像处理与内参")
    T = np.asarray(result["T_base_camera"], dtype=float)
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError(f"{side}: 外参不是有效刚体变换")

camera = results["left"]["camera"]
right_camera = results["right"]["camera"]
if camera["image_size"] != right_camera["image_size"]:
    raise ValueError("两次标定的左目图像尺寸不一致")
if not np.allclose(camera["K"], right_camera["K"], rtol=0, atol=1e-9):
    raise ValueError("两次标定的左目内参不一致")

combined = {
    "calibration_id": destination.stem,
    "reference_camera": "front_1",
    "extrinsics_convention": "T_base_camera",
    "extrinsics": {side: r["T_base_camera"] for side, r in results.items()},
    "intrinsics": {"front_1": camera["K"]},
    "image_sizes": {"front_1": camera["image_size"]},
    "source_results": {
        side: {
            "path": str(paths[side]),
            "sha256": hashlib.sha256(paths[side].read_bytes()).hexdigest(),
            "result": result,
        }
        for side, result in results.items()
    },
}
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")
print(destination)
PY
```

</details>

## 7. 采集真机数据

此命令连接并控制双臂。`uvc_rig.json` 已配置 LRCP `front_1`、`left_wrist`、`right_wrist`，
接口依次为 `5.3.2`、`5.3.3`、`5.3.4`，三路启动时均设置手动曝光 20 ms、亮度 48。
先按 [相机检查说明](egomimic/robot/piper_pico/README_uvc_front.md#1-相机配置与无机械臂录像检查)
确认三路图像和录像，再退出标定、模拟诊断和相机预览，保持 PCService / XRoboToolkit 正常运行。

```bash
python egomimic/scripts/collect_piper_pico.py \
  --left-can can0 --right-can can1 \
  --camera-config /home/yesen/pico_ego_probe/EgoVerse/calibrations/uvc_rig.json \
  --frequency 30 \
  --output /home/yesen/pico_ego_probe/EgoVerse/recordings/1106_pick
```

**B 开始/结束并保存一段；X 丢弃当前段；Y 双臂 home；A 保存当前段并退出。** Grip 离合、Trigger 夹爪。此模式使用 UVC 图像，不打开 PICO 视频流。

默认同时打开一个约 **960×304** 的三相机预览窗口，从左到右为左腕、front、右腕。
每路画面缩放到 320×240 区域并保持比例；预览刷新默认 15 fps，录制原图尺寸和相机/控制频率不变。
窗口由独立进程直接读取已有共享内存的最新帧，显示来不及时跳过旧帧，不额外打开相机，也不在遥操循环里显示。
`camera … fps` 是相机发布帧率，底部 `View …/15 fps` 是预览刷新率；`received … ms ago` 是距主机收到该帧的时间，**不是测得的相机延迟**。
超过 0.5 秒没有新画面会显示 `STALE`。

加 `--no-preview` 可关闭预览，加 `--preview-fps 10` 可降低预览刷新率（支持 1–30）。
点击窗口关闭按钮或在窗口按 Q/Esc **只关闭预览，采集继续**；collect 退出会自动关闭预览。
采集的 A/B/X/Y 仍使用 PICO 手柄，或先把焦点切回终端再按键；预览窗口不转发这些按键。
没有桌面显示时自动跳过预览，窗口启动失败也不阻止采集。预览与录制仍共享机器资源，帧率检查需分别看相机和控制循环。

仅测 front 可使用单 front 配置，但它不满足三相机训练配置。新增相机或改变配置后用新的录制目录。
每批重新设置 `RUN_NAME/ROBOT_RAW`，避免无意继续写入之前的批次。PICO 图像历史采集用法保留在
[采集说明](egomimic/robot/piper_pico/README.md#pico-图像采集历史可选模式)。

```text
$ROBOT_RAW/
  replay_buffer.zarr              # 逐控制拍的观测、命令与时间
  videos/
    cameras.json                 # 相机顺序
    camera_config.json           # 分辨率、延迟等
    0/
      0.mp4 / 0.npy              # front_1 与逐帧校正时间戳
      0.receive_timestamps.npy   # 原始主机接收时间
      1.mp4 / 1.npy              # left_wrist
      2.mp4 / 2.npy              # right_wrist
      robot0_state.npz
      robot1_state.npz
```

各路 UVC 请求帧率与控制 30 Hz 各自独立。实际帧率看时间戳，不能只看 MP4 标称播放速率。转换器使用已提交的 episode；失败时留下的零散视频不代表完整样本。

## 8. 转换真机数据

确认 `ROBOT_RAW` 是刚才那批原始录制、`CALIBRATION_JSON` 是第 6 节真实外参合并结果：

```bash
python -m egomimic.scripts.piper_pico_to_zarr \
  --input "$ROBOT_RAW" \
  --output-dir "$ROBOT_ZARR_ROOT" \
  --calibration "$CALIBRATION_JSON" \
  --width 640 --height 480 --undistort-alpha 0 \
  --fps 30 \
  --task-name pick_place \
  --task-description 'Pick up the object and place it in the container.'
```

按实际任务修改名称/描述。可加 `--episodes 0 2` 只转换指定段。输入是整批目录，不是 MP4；脚本没有单独的 `--intrinsics` 选项，K 从合并 JSON 读取。

```text
$ROBOT_ZARR_ROOT/原始目录名/
  episode_000000.zarr/
    conversion_report.json
  episode_000001.zarr/
    conversion_report.json
```

设置后续检查使用的路径（选择真实存在的 episode 编号）：

```bash
ROBOT_CONVERTED_BATCH="$ROBOT_ZARR_ROOT/$(basename "$ROBOT_RAW")"
ROBOT_EPISODE="$ROBOT_CONVERTED_BATCH/episode_000000.zarr"
```

当前转换输出 `piper_base_v3`，保留逐控制拍行，不重新采样或插值；`--fps 30` 不能将任意旧频率录制变成 30 Hz。
有原始 K/D 的 LRCP 图像先按保存的 K_new 去畸变，再按输出尺寸处理；未裁 ROI，输出 K 与图像一致。
没有畸变标定的历史视角保留原 resize 路径，不伪造腕部 K/D。不要提前手工缩放或去畸变两次。
每段保存左右 base 外参，机器人位姿仍保存在各自 base 系，由训练加载器变换到相机系。

同名 episode 不会覆盖。旧录制缺少逐拍字段可能被拒绝；`--allow-uncalibrated` / `--allow-missing-cameras` 只用于格式排查，不作为正式训练流程。

## 9. 导出和转换 PICO 人手数据

### 9.1 从头显拉取本地录制

这一路输入是 `CameraRecord_*.mp4 + trackingData_*.txt`，和机器人采集目录不同。先退出 PC 相机收流程序，在头显本地 Record 录制、停止后拉取：

```bash
adb shell 'ls -lt /sdcard/Download/CameraRecord_*.mp4 /sdcard/Download/trackingData_*.txt'

# 改成头显中实际文件的时间戳
record_stamp=20261029_213853
record_dir="$(mktemp -d "/home/yesen/pico_ego_probe/xrobo_record_${record_stamp}_XXXXXX")"
adb pull "/sdcard/Download/CameraRecord_${record_stamp}.mp4" "$record_dir/"
adb pull "/sdcard/Download/trackingData_${record_stamp}.txt" "$record_dir/"
```

### 9.2 用 10 月 10 日冻结 fit 转换

以下选用之前的 10 月 13 日录制作为可定位示例；处理新数据时将 `EGO_RAW` 改成刚拉取的 `record_dir`。

```bash
EGO_RAW="$PICO_WORKSPACE/xrobo_record_20261013_234134_zRntdg"
EGO_FIT="$PICO_WORKSPACE/xrobo_independent_fit_20261010_212402/fit.json"

python -m egomimic.scripts.pico_ego_to_zarr \
  --input "$EGO_RAW" --fit "$EGO_FIT" \
  --output-dir "$EGO_ZARR_ROOT" \
  --fps 30 --pose-sampling interpolate \
  --axis-convention piper_x --video-layout stereo-sbs

EGO_CONVERTED_BATCH="$EGO_ZARR_ROOT/$(basename "$EGO_RAW")"
EGO_EPISODE="$EGO_CONVERTED_BATCH/episode_000000.zarr"
```

使用冻结 fit，不在本段重拟合。`interpolate` 是默认位姿采样方式；需要最近邻版本可改为 `--pose-sampling nearest`。两种方式都应用 fit 中头/手时间偏移。双目拼接视频选 `stereo-sbs` 并使用左半幅；已经是单左目的视频改为 `--video-layout left`。

10 月 10 日 fit 是**人体 tracking 到相机的拟合参数**，不能代替 PiPER 的相机到 base 外参；复用前确认相机图像模式和几何仍匹配。输出为 `$EGO_CONVERTED_BATCH/episode_*.zarr`，连续有效片段可能拆成多段；这些切段仍属于同一次原始录制。

## 10. 可视化与训练输入检查

### 10.1 用户之前使用的网页浏览器

```bash
python egomimic/scripts/data_visualization/latent_inspector.py \
  --dataset-path /home/yesen/pico_ego_probe/egoverse_samples \
  --host 127.0.0.1 --port 8050
```

浏览器打开 <http://127.0.0.1:8050>。可将 `--dataset-path` 换成包含转换后 episode 的目录。当前这个通用浏览器对 Piper 身份的专用适配未完整注册；查看样例可以使用它，正式 PiPER 投影验收使用标定的 `verification/` 和下面的专用训练输入检查。

### 10.2 使用真实训练加载器检查图片、状态、动作

```bash
python -m egomimic.scripts.data_visualization.inspect_piper_pico_batch \
  "$ROBOT_EPISODE" --kind piper \
  --source-fps 30 --source-horizon 30 --action-stride 1 \
  --output-chunk-length 100 \
  --output "$EGOVERSE_REPO/diagnostics/${RUN_NAME}_batch_check_001"
```

```bash
python -m egomimic.scripts.data_visualization.inspect_piper_pico_batch \
  "$EGO_EPISODE" --kind pico_ego --gripper-mode zero \
  --expected-axis-convention piper_x \
  --source-fps 30 --source-horizon 30 --action-stride 1 \
  --output-chunk-length 100 \
  --output "$EGOVERSE_REPO/diagnostics/ego_batch_check_001"
```

输出目录必须是新的，包含 `frame_*.png` 与 `report.json`；显示训练加载器处理后、归一化之前的物理量。默认看开始/中间/结束，也可 `--indices 0 20 40` 指定有效样本。gripper 模式和时间参数要与训练一致。

### 10.3 人体投影视频 / 机器人三维轨迹

```bash
python -m egomimic.scripts.data_visualization.visualize_pico_ego_zarr \
  --input "$EGO_CONVERTED_BATCH" \
  --output-dir "$EGOVERSE_REPO/diagnostics/ego_projection_001"
```

输出叠加视频、样例、缩略图和数值摘要，不重拟合。PiPER 的 Rerun 回放：

```bash
python -m egomimic.scripts.data_visualization.visualize_piper_pico_zarr \
  --input "$ROBOT_CONVERTED_BATCH" --episode 0
```

保存为 `.rrd` 而不开窗口：

```bash
python -m egomimic.scripts.data_visualization.visualize_piper_pico_zarr \
  --input "$ROBOT_CONVERTED_BATCH" --episode 0 \
  --save "$EGOVERSE_REPO/${RUN_NAME}_episode_0.rrd"
```

这个 Rerun 工具不应用外参，轨迹分别在各臂 base 中；不要据此判断双臂共同空间关系或图像投影精度。

## 11. 模型训练与恢复

当前入口是 `train_piper_pico`，HPT + 共享 Flow Matching 动作头。机器人域使用三路图像和真实外参；人体域使用 `piper_x` 轴约定。启用验证时，每个所选数据域至少需要 **两个独立原始录制**，按原始录制分 train/validation，不能把同一录制的切段当成独立来源。

本机的 `converted_smoke` 使用过单位矩阵占位标定；10 月 25 日两份历史 `shared_fm_*` checkpoint 都只是 1 epoch / 4 batch 短测试。正式训练请用本手册真实标定后重新转换的数据，不使用这些测试产物。

### 11.1 只检查配置，不开始训练

```bash
python -m egomimic.trainHydra --config-name train_piper_pico \
  data=piper_pico_local \
  recordings.piper_root="$ROBOT_ZARR_ROOT" \
  recordings.pico_ego_root="$EGO_ZARR_ROOT" \
  recordings.gripper_mode=zero \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=32 \
  --cfg job
```

| 参数 | 选择 |
| --- | --- |
| `data=piper_local` | 只训练真机数据 |
| `data=pico_ego_local` | 只训练 PICO 人手数据 |
| `data=piper_pico_local` | 两域混合训练 |
| `recordings.gripper_mode=zero` | 人体域不使用夹爪宽度标签，公共动作头对应维填零 |
| `recordings.gripper_mode=width` | 人体域使用宽度标签 |

PiPER 动作始终是 14 维；`gripper_mode` 只改变人体域。两域数据根目录通过 `recordings.*_root` 指定，不是 `EGOVERSE_DATASET_DIR`。

### 11.2 先跑一次短流程检查

```bash
CUDA_VISIBLE_DEVICES=0 python -m egomimic.trainHydra --config-name train_piper_pico \
  data=piper_pico_local \
  recordings.piper_root="$ROBOT_ZARR_ROOT" \
  recordings.pico_ego_root="$EGO_ZARR_ROOT" \
  recordings.gripper_mode=zero \
  recordings.batch_size=4 recordings.num_workers=2 \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=32 \
  trainer.min_epochs=1 trainer.max_epochs=1 \
  trainer.limit_train_batches=4 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=1 \
  callbacks.model_checkpoint.every_n_epochs=1 \
  description=smoke
```

此步骤只检查数据/设备/保存流程，产生的权重不能视为已训练好的策略。只使用真机数据时改为 `data=piper_local`，人体数据根参数可省略。

### 11.3 完整训练模板

下面 200 epoch、batch 16 是可修改的实验起点，不是已经选定的最优参数：

```bash
CUDA_VISIBLE_DEVICES=0 python -m egomimic.trainHydra --config-name train_piper_pico \
  data=piper_pico_local \
  recordings.piper_root="$ROBOT_ZARR_ROOT" \
  recordings.pico_ego_root="$EGO_ZARR_ROOT" \
  recordings.gripper_mode=zero \
  recordings.batch_size=16 recordings.num_workers=4 \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=32 \
  trainer.min_epochs=1 trainer.max_epochs=200 \
  trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 \
  trainer.check_val_every_n_epoch=1 \
  callbacks.model_checkpoint.every_n_epochs=10 \
  model.scheduler.T_max=200 \
  description=real_calibrated
```

`1.0` 表示完整 loader 的比例；整数 `1` 表示只用一个 batch。默认配置原本是 CPU、2000 epoch、每轮限 100/80 个 train/validation batch；因此上面显式指定 GPU、训练长度和完整 loader。混合训练中较短的数据域会循环到较长域结束。显存不足优先减小 batch。首次使用视觉编码器可能下载预训练权重。

当前默认时间设置：

```text
recordings.piper_timing.source_fps=30
recordings.piper_timing.source_horizon=30
recordings.piper_timing.action_stride=1
recordings.ego_timing.source_fps=30
recordings.ego_timing.source_horizon=30
recordings.ego_timing.action_stride=1
recordings.output_chunk_length=100
```

30 个点包含当前时刻，覆盖约 `29/30 = 0.967 s`；100 是该跨度上的输出点数，不是机械臂 100 Hz 控制。不要把 PICO 相机 60 fps 写成机器人控制数据的 source_fps。

默认训练输出：

```text
$EGOVERSE_REPO/logs/piper_pico/<description>_<时间>/
  .hydra/config.yaml
  .hydra/overrides.yaml
  trainHydra.log
  checkpoints/last.ckpt
  checkpoints/epoch_*.ckpt
  norm_stats/norm_stats.json
  videos/
```

可用 `EGOVERSE_LOG_DIR` 改日志根目录。恢复训练时在**原来同一组训练命令末尾**加 `ckpt_path=/绝对路径/checkpoints/last.ckpt`；`trainer.max_epochs` 是累计目标轮数。保留数据域、时间设置、gripper 模式和模型结构，checkpoint 内统计量/数据契约会被恢复校验。

### 11.4 Pi0.5 JAX 独立训练与服务

完整命令见 [Pi0.5 JAX 工作流](pi05/README.md)，包括用户自行配置环境、转换器导出、100 点插值、归一化、LoRA 训练/恢复测试、离线推理和 server/client 真机入口。

Pi0.5 保留 **100 点训练动作**。原 Zarr 仍按采集 tick 保存，训练加载窗口时才插值到 100 点；默认 30 Hz、30 个源动作的首末时间跨度仍为 0.966667 秒。推理沿用时间重采样后按 30 Hz 执行，不能把 100 点直接逐点按 30 Hz 下发。

新转换可在第 8 节命令添加 `--pi05-manifest /绝对路径/manifest.json`；已有转换数据使用 `python -m egomimic.scripts.export_piper_pi05`，无需重新处理视频。环境由用户配置，机器人端继续使用 `emimic`；模型端使用独立 JAX Python。

## 12. 模型推理与真机轨迹回放

### 12.1 模型离线推理，不运动机器人

`replay_piper_policy` 离线 CPU 推理入口读取转换后的 episode 和 checkpoint。实际三相机、双臂反馈的实时入口见 **12.3**，两者参数和报告格式各自独立。

先设置权重、episode，以及经确认的两臂各自 base 工作空间边界（米）。下列 12 个范围变量需要自行填写，不能用数据集 min/max 自动代替：

```bash
POLICY_CHECKPOINT=/绝对路径/可信训练权重.ckpt
# 先设置 LEFT_MIN_X/Y/Z、RIGHT_MIN_X/Y/Z、LEFT_MAX_X/Y/Z、RIGHT_MAX_X/Y/Z
python -m egomimic.scripts.replay_piper_policy \
  --checkpoint "$POLICY_CHECKPOINT" --episode "$ROBOT_EPISODE" \
  --output-dir "$EGOVERSE_REPO/diagnostics/policy_replay_001" \
  --workspace-min "${LEFT_MIN_X:?请设置左臂边界}" "${LEFT_MIN_Y:?}" "${LEFT_MIN_Z:?}" \
                  "${RIGHT_MIN_X:?请设置右臂边界}" "${RIGHT_MIN_Y:?}" "${RIGHT_MIN_Z:?}" \
  --workspace-max "${LEFT_MAX_X:?}" "${LEFT_MAX_Y:?}" "${LEFT_MAX_Z:?}" \
                  "${RIGHT_MAX_X:?}" "${RIGHT_MAX_Y:?}" "${RIGHT_MAX_Z:?}" \
  --inference-steps 10 --control-frequency 30 --execute-steps 1 --max-chunks 1
```

输出 `report.json`、`scheduled_actions.jsonl`、`sent_actions.jsonl`，重跑时换一个新的输出目录。这里 sent 仅表示交给离线记录器，没有向 CAN 发命令；虚拟时间回放也不能证明真机实时速度。

### 12.2 回放录制时的命令，不使用模型

先离线检查：

```bash
python -m egomimic.scripts.replay_piper_dataset \
  --input "$ROBOT_CONVERTED_BATCH" --dry-run
```

确认需要实际回放、已退出所有其它机械臂控制程序后，下面才会运动真机：

```bash
python -m egomimic.scripts.replay_piper_dataset \
  --input "$ROBOT_CONVERTED_BATCH" \
  --left-can "$LEFT_CAN" --right-can "$RIGHT_CAN" \
  --approach-duration 4 --speed 0.5 --execute
```

`speed=0.5` 按原间隔两倍时间回放，不是 SDK 速度百分比。先从实测当前状态渐变到该段第一条 command，再进入回放，不是统一回 home。此工具读取各自 base 系的录制命令，不需要 PCService、PICO 或视频；输入必须为转换后的 PiPER 数据，不能传人体数据或原始 `replay_buffer.zarr`。报告默认在 `diagnostics/piper_dataset_replay_*.json`。

### 12.3 实时同步模型推理（三 UVC + 双臂，真机）

入口 `infer_piper_policy` 使用终端键盘控制，不需要 PICO 头显或 `runService.sh`。三台 UVC 及左右 CAN 使用当前配置，启动前退出占用相机或控制双臂的采集、遥操作、回放程序。

先运行只观察模式：读取实际图像与机械臂反馈、进行模型预测并记录候选动作；不使能机械臂、不回 home、不发送机械臂或夹爪目标。它仍需要三台相机和两臂反馈在线。

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

实际执行用下面这条命令。`--execute` 允许下发动作；`--home-on-start true` 会在预热后回到采集时的双臂 home，**同时打开夹爪**，确认反馈到位后等待按键开始。若希望从当前位置开始，将它改为 `false`。只观察模式不能同时指定 `--home-on-start true`。

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

在**终端**按空格开始/暂停，`P` 暂停，`Q` / Esc / Ctrl+C 退出。启动预热只预测一次、不执行；预热或回 home 期间输入的开始键不会自动启动策略。暂停丢弃剩余预测，恢复时重新取图像和反馈。预览窗口中的 `Q` / Esc **只关闭预览**；推理控制键必须在终端输入。

本次 checkpoint 的 100 点覆盖约 0.967 秒，程序从 checkpoint 的训练时间设置推导，恢复为 30 点后按目标 30 Hz 执行，默认每轮执行 30 点再重新观察。流程同步进行，不补偿或跳过模型计算占用的时间，所以跨轮次间隔可能变长；日志记录实际推理耗时与提交间隔。相机仍由独立进程采集，预览从 SHM 读取三路画面，默认 15 fps；加 `--no-preview` 可关闭。

运行日志默认开启，显式开启用 `--log`；在原命令末尾加 `--no-log` 可关闭逐动作、chunk 和事件日志，终端状态/错误以及启动配置、结束摘要仍保留。开启时由独立进程编码 JSON 和写盘，动作循环通过有界队列非阻塞提交。队列满则丢弃新日志、写入故障则提示记录不完整；结束摘要 `logging` 给出提交/写入/丢弃数量和错误。检查卡顿日志前先确认 `logging.incomplete=false`，否则不能假定各文件完整、逐行对应。此开关不影响离线评估报告。

若动作调度迟到超过 100 ms，程序在双臂保持成功后暂停，丢弃剩余动作；终端空格用新图像和反馈重新推理，`Q` 退出。不会自动继续旧段或集中补发。反馈失效、下发失败、时钟异常、保持失败仍停止程序。开启日志时，`events.jsonl` 的 `schedule_paused` 和 `cycles.jsonl` 的分段耗时用于定位卡顿。

四个路径参数均必填。预检将 `--calibration` 的内外参与 `--training-data` 中所有选中 PiPER episode 的图像处理、夹爪编码、相机配置核对；若目录包含多个机器人几何配置，应选择当前 rig 的子目录或单个 episode。旧 checkpoint 本身没有保存实际标定 SHA，这一步通过指定的训练数据核对。当前模型用单位外参训练，**不能只替换为真实外参 JSON 就继续使用该模型**；后续用真实外参转换数据、重新训练后，使用同一推理入口即可。

输出默认在 `diagnostics/piper_live_<UTC时间>_<随机后缀>/`，包括 `arguments.json`、`manifest.json`、`summary.json`，以及开启日志时的 `cycles.jsonl`、`events.jsonl`、`commands.jsonl`。`commands` 的实际执行记录表示两臂 SDK 提交确认，不表示机械臂已经物理到位。可用 `--output-dir /绝对路径/新目录` 指定位置。该入口尚未完成真实硬件上的策略运行验收；详细配置与报告字段见 [实时推理说明](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_live_policy.md)。

### 12.4 用训练集验证推理 pipeline 和动作误差（完全离线）

此命令经过同一个 `infer_piper_policy` 入口，不打开相机或 CAN，不移动机械臂。CPU 推理，输入训练时的已处理 RGB 和实际 obs state；训练 command 只用于算误差，不输入模型。

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

默认所有集，每集均匀抽 8 个完整窗口；加 `--eval-max-episodes 2 --eval-samples-per-episode 2` 可以快速检查。默认结果存到新的 `diagnostics/piper_offline_eval_<UTC>_<id>/`，包含 `report.md`、`summary.json`、按预测时间的误差曲线，以及 `plots/` 中的三路输入与轨迹对比图。

报告分别给出左右臂位置（mm）、旋转（°）、夹爪（mm）误差，并对照保持当前状态的基线。输入/坐标/输出一致性通过与模型误差大小分别报告；训练集误差不代表真机闭环成功率。详细解释见 [离线验证说明](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_live_policy.md)。

## 13. 历史命令与详细说明

### 13.1 旧的纯 Controller Pose 检查

这是另一个项目环境，只打印控制器/头显与按钮，不连接 PiPER。需要 PCService：

```bash
cd /home/yesen/cys/avfrl_real
source /home/yesen/cys/avfrl_real/.venv/bin/activate
python third_party/Xense-Pico-Teleop-Interface/examples/pico4_controllers.py
```

结束后，回到本手册的环境：

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source /home/yesen/pico_ego_probe/EgoVerse/emimic/bin/activate
```

### 13.2 10 月 13 日那一段的历史投影复现

这个脚本内部写死了对应数据与输出路径，只用于复现旧结果，会写回该历史输出目录；处理新数据请用第 9 节通用转换入口。

```bash
/home/yesen/cys/hand-6dof-pipeline/.venv/bin/python -B \
  /home/yesen/pico_ego_probe/xrobo_apply_fit_20261010_20261013_234134/apply_fit_20261010.py
```

历史输出目录：`/home/yesen/pico_ego_probe/xrobo_apply_fit_20261010_20261013_234134/`，包含 `overlay/stereo_hands.mp4`、样例图、`summary.json` 和 `report_zh.txt`。

### 13.3 查看完整参数

```bash
python egomimic/scripts/collect_piper_pico.py --help
python egomimic/scripts/calibrate_piper_pico.py collect --help
python egomimic/scripts/calibrate_piper_pico.py solve --help
python egomimic/scripts/calibrate_camera_latency.py --help
python -m egomimic.scripts.piper_pico_to_zarr --help
python -m egomimic.scripts.pico_ego_to_zarr --help
python -m egomimic.scripts.replay_piper_policy --help
python -m egomimic.scripts.infer_piper_policy --help
python -m egomimic.scripts.replay_piper_dataset --help
```

详细设计和参数说明：

- [真机多相机采集](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README.md)
- [相机延迟测量](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_camera_latency.md)
- [外参标定与 SDK 末端图片](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_calibration.md)
- [PiPER 数据格式与转换](/home/yesen/pico_ego_probe/EgoVerse/egomimic/scripts/README_piper_pico_to_zarr.md)
- [PICO 人手数据转换](/home/yesen/pico_ego_probe/EgoVerse/egomimic/scripts/README_pico_ego_to_zarr.md)
- [当前训练流程](/home/yesen/pico_ego_probe/EgoVerse/egomimic/rldb/README_piper_pico_training.md)
- [时间与数据契约](/home/yesen/pico_ego_probe/EgoVerse/egomimic/rldb/README_piper_pico_stage1.md)
- [模型离线推理](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_policy_inference.md)
- [模型实时推理](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_live_policy.md)
- [录制轨迹回放](/home/yesen/pico_ego_probe/EgoVerse/egomimic/robot/piper_pico/README_dataset_replay.md)

旧转换文档中“专用训练加载器未接入”等描述已过时；训练可用性以当前训练文档和代码为准。

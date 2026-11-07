# 固定 UVC / PICO 左目与 PiPER 双臂的眼在手外标定

独立入口为 `egomimic/scripts/calibrate_piper_pico.py`，提供 `collect`、
`solve`、`check-image` 和 `export`。每次只标定一只手臂，两个会话必须使用
同一台固定相机。旧 PICO 路径使用**左目**；UVC 路径从 `camera_rig_v1` 选择
已标定原始内参的相机。原采集入口的按键和录像流程保持原样。

相机和双臂底座在标定、正式采集及推理时必须保持相对固定。将 ChArUco
板牢固夹在本次标定手臂上；一个会话内不能重新夹持、松动或调整板的位置。
重新夹持后使用新会话，不要把前后样本混在一起。

## UVC 原始图像路径

`--camera-rig rig.json --camera-name front_1` 使用采集器同一份逐相机配置，
其中 `front_1` 必须为 UVC 并提供 `calibration_file`。配置加载器会保存原始
内参 JSON 的 K、非零 D、原始尺寸、相机模型和来源 SHA-256，检查尺寸与对焦
条件一致；不会缩放、镜像或去畸变图像。无需 `--headset-ip` 接收图像，
PICO 手柄遥操作仍通过现有 `PicoProcess` 运行。

UVC 路径不能与 `--camera-config`、`--headset-ip` 或非默认 PICO 图像参数混用。
`--camera-config` 仍表示旧 PICO 单目内参 JSON。稳定时间检查使用所选 UVC 的
`latency`；`latency_status=unmeasured` 只表示未测量，不能当作已测延迟。

离线图像检查不连接任何设备：

```bash
python egomimic/scripts/calibrate_piper_pico.py check-image \
  --camera-rig rig.json --camera-name front_1 \
  --image original_front.png --output front_check
```

准备好实际硬件后，以下采集命令会连接并使能双臂：

```bash
python egomimic/scripts/calibrate_piper_pico.py collect \
  --arm left --camera-rig rig.json --camera-name front_1 \
  --left-can can0 --right-can can1 --output uvc_left_session
```

右臂使用 `--arm right --output uvc_right_session`。两边分别用现有 `solve`
子命令求解。结果保留真实相机类型、名称、模式和完整原始内参；UVC 结果不会
标成 `camera_eye=left`。求解成功并具有至少 3 个留出验证样本后，可离线合并：

```bash
python egomimic/scripts/calibrate_piper_pico.py export \
  --camera-rig rig.json --camera-name front_1 \
  --left-result uvc_left_session/calibration_result.json \
  --right-result uvc_right_session/calibration_result.json \
  --calibration-id my-fixed-uvc-rig --output calibration.json
```

导出会核对左右角色、`T_base_camera` 方向、米制单位、原始内参来源与几何、
相机身份、采集模式及对焦设置，拒绝已有输出文件。导出文件标记
`image_geometry=raw_distorted`，包含原始 K/D 与真实左右外参，不包含假定外参。
内参与外参的质量警告分开保留，`production_approved` 不会自动变成 true。
这是原始图像标定；供仅支持针孔模型的训练使用前，离线转换必须去畸变图像并
同步更新 K，不能把原始非零 D 静默丢弃。

## 参数与实际打印尺寸

默认板参数对应实测缩小到 95% 的打印件：6×6 格、方格 **19 mm**、
标记 **13.3 mm**、整幅网格 **114×114 mm**，字典 `DICT_5X5_100`，
`legacyPattern=False`。13.3 mm 假定标记随纸张等比例缩小；使用前应复核实物。
不要沿用 PDF 标称的 20 mm。尺寸改变后重新求解，不能直接缩放已求出的
相机到机器人底座矩阵。

默认左目图像为 1080×810，内参沿用此前验证的值：

```text
K = [[686.8340695362958, 0, 539.75],
     [0, 686.8680648828482, 404.5],
     [0, 0, 1]]
D = 0
```

这里的零畸变是现有输出图像的使用假设，不代表原始传感器没有镜头畸变。
这组 K 已是单眼参数，不要再将横向参数除以 2。相机模式、分辨率或图像处理
发生变化时，应重新验证。可通过 `--camera-config` 和 `--board-config` 指定
配置 JSON；运行时实际采用的配置会保存到会话中。

相机 JSON 示例（`image_size` 顺序为宽、高）：

```json
{
  "eye": "left",
  "image_size": [1080, 810],
  "K": [[686.8340695362958, 0, 539.75], [0, 686.8680648828482, 404.5], [0, 0, 1]],
  "D": [0, 0, 0, 0, 0]
}
```

标定板 JSON 示例（长度单位为米）：

```json
{
  "squares_x": 6,
  "squares_y": 6,
  "square_length_m": 0.019,
  "marker_length_m": 0.0133,
  "dictionary": "DICT_5X5_100",
  "legacy_pattern": false
}
```

## 采集

先按原遥操作流程启动 PCService 和头显 XRoboToolkit，准备好两路 CAN。
关闭头显本地 Record、其它相机预览以及正在运行的原采集或推理程序。
下例 IP 仅为占位值，替换为本机实际地址；命令会连接并使能真实机械臂。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

python egomimic/scripts/calibrate_piper_pico.py collect \
  --arm left \
  --headset-ip 192.168.1.9 \
  --pc-ip 192.168.1.8 \
  --left-can can0 --right-can can1 \
  --output calibration_left_session
```

标定右臂时改为 `--arm right`，并使用新的输出目录。省略 `--output` 会自动
建立独立会话目录；已有目录不会继续追加。`--arm` 只决定本次接受哪只手臂的样本，双臂遥操作保持
原有 Grip 离合、Trigger 夹爪、映射和滤波行为。

| 按键 | 标定脚本中的作用 |
| --- | --- |
| X | 左臂回 home |
| Y | 请求记录左臂姿态和左目图像；仅 `--arm left` 接受 |
| A | 右臂回 home |
| B | 请求记录右臂姿态和左目图像；仅 `--arm right` 接受 |
| V（电脑键盘） | 切换下一次采样为 TRAIN 或 VALIDATION |
| U（电脑键盘） | 作废最近一个有效样本，保留原始文件 |
| Q / Esc / Ctrl+C / 关闭预览窗口 | 退出，保留已保存样本 |

按错另一只手的记录键会提示，不会保存。手柄按键只响应按下边沿，长按不会
重复采集。已有请求未完成时，新请求会提示忙碌，不积压队列。

回 home 只移动对应手臂，延续其此前有效夹爪命令；期间按原有 home 流程
暂停双臂遥操更新，默认 3 秒后重新建立离合参考。回 home 会取消尚未通过检查的
采样；已经通过检查并开始写盘的快照仍可完成保存。恢复遥操后 Trigger 仍按原逻辑控制夹爪，**保持夹持所需的 Trigger
输入**；松开 Grip 并不会冻结夹爪。

记录键触发后，程序等待目标臂稳定并检查图像和实际反馈持续更新，然后
**先固定同一份图像与实际末端 pose，再后台检测该图，最后保存这对数据**。
直接读取现有相机和机器人进程的缓存，不调用 `get_obs()`，不插值位姿，
不另建相机或 CAN 连接。保存的是 SDK 实际反馈，不是手柄姿态或下发目标。

这仍不是硬件同步曝光。静止窗口用于降低图像传输延迟造成的配对误差；
程序同时保留原始时间戳、延迟参数和反馈来源新鲜度，拒绝陈旧帧、停更反馈、
运动中的样本以及检测失败的样本。图像与 pose 固定后不会因检测耗时而重新
读取 pose。采样分类在按记录键时固定，随后按 V 不会改变正在处理的样本。
SDK 的末端时间戳代表最近更新的 CAN 分包，并不能证明六个分量在同一时刻
更新，因此仍需上述静止检查。

默认电脑预览显示板检测、角点数、重投影 RMS、等待/拒绝原因和已保存数量。
反馈显示在电脑窗口和终端，不发送头显震动。无图形环境可加 `--no-preview`；
终端快捷键需要交互式终端。计数仅在样本保存成功后增加。

主要默认门限可在 `collect --help` 中调整：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--stable-seconds` | 0.7 | 采样前稳定窗口，秒 |
| `--post-stable-seconds` | 0.15 | 快照后稳定检查，秒 |
| `--capture-timeout` | 3 | 单次请求最长等待，秒 |
| `--position-tolerance-mm` | 1 | 稳定窗口内位置变化门限，毫米 |
| `--rotation-tolerance-deg` | 0.3 | 姿态变化门限，度 |
| `--gripper-tolerance-mm` | 0.5 | 夹爪变化门限，毫米 |
| `--max-pose-age` | 0.15 | 反馈新鲜度门限，秒 |
| `--max-camera-age` | 0.3 | 图像新鲜度门限，秒 |
| `--min-corners` | 16 | 最少 ChArUco 内角点，总数 25 |
| `--max-rms-px` | 0.8 | 单帧重投影 RMS 上限，像素 |
| `--pico-camera-latency` | 0.1 | 图像接收时间的延迟补偿，秒 |

前后稳定窗口总长最多 1 秒，采样超时必须大于该总长；前稳定窗口须大于
相机延迟补偿与最大反馈年龄之和。
这些是采样筛选门限，不是承诺的最终毫米精度。保持板清晰、平整、无遮挡，
采集约 **30–40 个 TRAIN 姿态和 8–10 个 VALIDATION 姿态**；覆盖不同位置、
距离和倾角，绕不止一个旋转轴改变姿态。仅平移或只绕单轴转动不足以可靠
求解。验证样本应是新的姿态，并且板在同一夹持状态下，不能重夹后补拍。

## 会话文件与离线求解

```text
calibration_left_session/
  session.json                 # 手臂、相机、标定板及运行配置
  events.jsonl                 # 操作、拒绝原因等事件
  runtime/                     # 复用环境生成的空录像结构及相机配置
  samples/
    000001/
      raw.png                  # 未画标注的左目图像
      overlay.jpg              # 检测预览
      sample.json              # 分类、实测姿态、时间、检测和质量结果
      feedback.npz             # 采样附近的实际反馈
      invalidated.json         # 仅在该样本被作废后出现
```

原始文件保留用于复核；作废样本不参与求解。检测及磁盘写入放在后台，避免
阻塞遥操作循环；样本目录提交完整后才视为成功。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

python egomimic/scripts/calibrate_piper_pico.py solve \
  --session calibration_left_session \
  --output calibration_left_result.json
```

求解只使用 TRAIN 样本，至少需要 8 个有效训练姿态，并检查旋转多样性。
VALIDATION 姿态不参与拟合；至少 3 个才将留出验证标记为可用，实际建议 8–10 个。
样本数达到门限不代表姿态分布合格。求解使用 SciPy 联合优化，不依赖当前
OpenCV 环境缺少的 `calibrateHandEye`，也不会修改正式采集和推理配置。

约定 `T_A_B` 将 B 坐标变换为 A 坐标。每帧满足：

```text
T_base_camera @ T_camera_board = T_base_end @ T_end_board
p_base = T_base_camera @ p_camera
```

输出 `T_base_camera`、其逆矩阵、固定的 `T_end_board` 及训练/验证结果；
平移单位为米。末端坐标系使用 SDK 实际反馈的 end 坐标系，不默认为夹爪
指尖。无需预先测量板到末端的安装变换，但它必须在整个会话保持固定。
分别检查左右臂结果后，再将两个 `T_base_camera` 接入正式采集及推理。
省略 `--output` 时，结果写入会话目录的 `calibration_result.json`。

每次 `solve` 成功后会**自动生成验收图片**，终端打印完整目录，无需另运行
可视化脚本。使用 SDK 实际返回的末端坐标系，标记为 `L SDK_END` 或
`R SDK_END`，不添加夹爪 TCP 偏移。坐标轴默认长 40 mm，X 红、Y 绿、Z 蓝。

```text
calibration_left_session/
  calibration_result.json
  verification/
    时间戳_唯一编号/
      README.txt
      calibration_used.json       # 此批图片对应的标定结果，重跑后仍保留
      summary.json                # 逐样本投影与角点误差、缺失图片原因
      validation/                 # 优先检查：未参与拟合的样本
        000033_sdk_end.png         # 原图 + SDK 末端三轴
        000033_board_check.png     # 三轴 + 角点预测/检测/误差连线
        contact_001.jpg            # 缩略总览；每页最多 6 个样本
      train/                      # 参与拟合的样本，同样生成两类图片
```

两类 PNG 都保持原图分辨率，说明栏添加在图像之外。验收图中的标定板预测
来自 `inverse(T_base_camera) @ T_base_end @ T_end_board`，不是把该帧图像 PnP
结果直接画回去。检测角点与预测角点用不同符号和颜色表示，连线显示偏差；
此处的角点 RMS 衡量完整标定链路，不等同于采样时单帧 PnP 的 RMS。
末端位于画面外或相机后方时会明确标注，避免出现误导性的坐标轴。

每次求解生成新图片目录，保留旧图；图片目录内的 `calibration_used.json`
固定保存这一次使用的结果。若需要指定图片位置或调整轴长，在 `solve` 后加
`--visualization-dir /absolute/path/to/new_verification_dir --axis-length-mm 50`。
指定目录必须不存在。即使 `--output` 将外参 JSON 放在别处，默认图片仍放在
会话的 `verification/` 中。

被作废、未提交或检测不合格的样本不会出图。原图丢失、损坏或尺寸不符时，
程序保留成功求出的外参数值，并在终端及 `summary.json` 明确报告未生成的
图片；不会用缩放图或空白图代替真实照片。原始采样图片保持不变。

低重投影误差只说明图像拟合较好，不能单独证明机器人空间精度。还应检查
留出姿态的一致性，并使用独立物理测量验证所需精度。相机、底座移动后重标；
单独重夹标定板不影响已经完成的相机到 base 外参，但旧、新夹持样本不能混合。

## 离线检查一张图

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

python egomimic/scripts/calibrate_piper_pico.py check-image \
  --image /absolute/path/to/left_1080x810.png \
  --output calibration_image_check
```

此命令使用同一套 ChArUco 参数和质量判断，只读取图片，不连接机器人或相机。
输出 `detection.json` 和 `overlay.jpg`；输出目录必须是新目录。
输入应为与 K 匹配的单眼原图，不能直接传拼接双目图或任意缩放后的预览。
它可以检查检测和板到相机的位姿；没有配对的机器人实际反馈，手持板视频
无法用于求解相机到机器人 base 的外参。

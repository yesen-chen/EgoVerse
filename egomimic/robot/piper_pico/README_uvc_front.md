# UVC front_1 + UVC 双腕：实施和验收

入口仍是 `egomimic.scripts.collect_piper_pico`。PICO 只提供手柄姿态；正式遥操和外参采集
仍需运行 PCService，并在 XRoboToolkit 中连接电脑。相机独立检查不需要这项服务。不提供
`--headset-ip` 就不会打开它的图像流。三路 UVC 的采集、录像、时间戳、控制行
保存继续使用原环境。下面均从仓库根目录执行，先 `source emimic/bin/activate`。

## 1. 相机配置与无机械臂录像检查

本机三相机配置已更新为 `calibrations/uvc_rig.json`；采集时显式传入该文件。
单 front 测试和标定使用 `calibrations/uvc_front_lrcp_480p_002_camera.json`，两份配置的 front 设置一致。
程序不会在省略参数时自动加载 rig。旧 KD 前置相机已损坏，当前 `front_1`
改为 LRCP F1080P；旧相机的内参、外参和延迟结果只供历史数据复现。
新机器可参考 `egomimic/robot/piper_pico/configs/uvc_three_camera.example.json` 建立自己的 rig。

2026-11-05 三台接齐后的角色已由现场确认。下表的节点编号只是当时的枚举结果；
配置使用 `/dev/v4l/by-path/pci-0000:0f:00.0-usb-0:<接口>:1.0-video-index0`：

| 角色 | USB 接口 | 当时节点 | VID:PID | USB 设备版本 |
| --- | --- | --- | --- | --- |
| `front_1` | `5.3.2` | `/dev/video0` | `1bcf:22c8` | `0039` |
| `left_wrist` | `5.3.3` | `/dev/video4` | `1bcf:22c8` | `0039` |
| `right_wrist` | `5.3.4` | `/dev/video2` | `1bcf:2cc8` | `0031` |

三台都上报序列号 `000000000001`，已经出现 `by-id` 名称冲突。
front 与左腕即使视场不同，现有 USB 标识仍完全相同，不能可靠地按硬件编号自动区分。
因此保留 `by-path` 和 `expected_identity` 检查；重连时使用相同插口最省事。
更换接口时同时更新 `device` 和 `expected_identity.ID_PATH`，原始标定中的历史接口记录保留。
front 与左腕互换插口无法通过这些硬件标识发现，换线后仍需预览核对视角。

- `front_1` 使用已标定的 **640×480、30 fps、MJPG**；两腕也分别按自己的配置打开。
- `controls` 使用 `v4l2-ctl` 的控制名称和整数单位。只填设备支持的项。
- 三台正式统一为 `brightness=48`、`auto_exposure=1`、`exposure_time_absolute=200`，
  即亮度 48、手动曝光 20 ms。采集启动会应用这些值，不依赖硬件保留状态。
  每台相机可在 `controls` 中单独设置
  `"brightness": 40` 等整数覆盖默认值，未填则使用 48；其它控制项仍保留。
  旧 `--camera name=/dev/videoN` 方式可用 `--camera-brightness 40` 统一设置。
  使用 `--camera-config` 时以 JSON 为准。启动会设置并读回核验，设备不支持
  或读回不一致会报错；参数会保存在录制元数据中。亮度使用驱动单位，不是曝光时间。
  新默认值与旧录制配置不同时，请使用新输出目录，避免混用不同设置的数据。
- front 的 `calibration_file` 指向原图 `calibration.json`，相对路径以 rig JSON
  所在目录为基准。当前 front 使用
  `uvc_front_lrcp_480p_002_result/calibration.json`。
  **不要传 `rectified_camera.json`**。文件中的质量警告仍会保留。
- 当前 LRCP 没有软件对焦控制，保持镜头固定；不要沿用旧 KD 的 `focus_absolute=0`
  等控制项。已知相机身份和几何控制设置会在运行时核对，未知项不会冒充“已验证”。
- 每路 `latency` 独立填写秒数，`latency_status` 为 `unmeasured` 或 `measured`。
  本次按用户决定不重测延迟，继续使用历史补偿：front 为三次测试合并均值 **0.067275 s（约 67.3 ms）**。226 个有效二维码的
  中位数为 67.107 ms、P95 为 85.188 ms；包含未独立测量的屏幕显示延迟，且只验证了
  单相机录像负载。两腕保留 **0.066 s**；右腕来自此前测量，左腕沿用右腕值、尚未独立测量。
  两腕不套用 front 的结果。配置记录历史沿用来源，新曝光下标为未测；不表示补偿被置零。
  通用样例的 0 是待测占位值。
- 配置文件与旧 `--camera` / `--headset-ip` 不能同时传；其余相机模式参数由配置决定。

配置和角色映射已完成，三台 20 ms、亮度 48 的设置已从预览正式采用。
15 秒三路录像通过，front/左腕/右腕分别为 **30.018/30.015/30.013 fps**，
没有超过检查阈值的帧间隔。报告在 `diagnostics/uvc_all_20ms_brightness48_preview_001/capture/report.json`。
该检查验证录像与帧率，没有重新测量延迟。

此前 2026-11-05 的 10 秒三路实测中，设备身份、640×480 图像、
亮度读回、录像和关闭正常，但 front 平均只有 22.06 fps，两腕分别为 30.01/30.00 fps，
因此帧率验收未通过。随后同场景单独录 front 仍只有 20.24 fps；不能把降速归因于三路并发。
随后进行了三路同开的曝光对照：front 自动曝光为 20.22 fps，固定 20 ms 为 30.00 fps，
切回自动曝光为 25.41 fps，固定 30 ms 为 30.01 fps。固定 20 ms 的 30 秒复测三路均约
30.01 fps、全部通过。证据确认当前条件下自动曝光会降帧；并不表示自动模式在任何场景都降帧。
固定 30 ms 那组曾有一次三路共同停顿，双腕超出间隔阈值；该组不能称为整体通过。
实验汇总在 `diagnostics/uvc_exposure_ab_20261105_001/summary.json`，复测在同目录
`confirmation_20ms.json`。这次早期对照时，双腕读回为手动曝光（`auto_exposure=1`）：
左腕 `exposure_time_absolute=200`（20 ms）、右腕 `300`（30 ms），当时 rig 尚未固定腕部曝光。
随后已将三台统一为手动 20 ms、亮度 48，并写入当前正式配置。
报告在 `diagnostics/uvc_lrcp_replacement_20261105_3cam_001/report.json` 和
`diagnostics/uvc_lrcp_replacement_20261105_front_only_001/report.json`。
67.275 ms 来自此前自动曝光、30 fps 的二维码测试，不能当作手动曝光下已经复测的延迟。

后续复查命令如下；只打开相机，不启动 PICO 姿态服务或机械臂，输出目录须使用新名字：

```bash
python -m egomimic.scripts.check_camera_rig \
  --camera-config calibrations/uvc_rig.json \
  --duration 60 --cycles 3 \
  --output diagnostics/uvc_3cam_hardware_001
```

每轮重新创建、打开、录像、关闭所有相机。报告检查实际解码帧数、原始尺寸、
接收时间戳、补偿后时间戳、录像时长、新鲜度、帧间隔和正常退出。
默认实测帧率至少达到请求值的 90%，最大间隔不超过三个请求帧周期。
驱动声明的 30 fps 不等于录像实测 30 fps；主机到帧差也不证明曝光同步。

录制新增 `N.receive_timestamps.npy` 和 `N.stream_metadata.json`，原来的
`N.npy` 仍为接收时间减配置延迟。原始 MP4 不去畸变、不裁剪、不缩放。
焦距/控制核验在启动和录像边界进行，并非连续监控所有中途设置变动。

## 2. 同一配置测延迟及外参

本次沿用历史延迟补偿，不要求再次测量。以下命令保留供按需复核；测试时复用同一台相机、分辨率、FourCC、焦距和曝光参数：

```bash
python -m egomimic.scripts.calibrate_camera_latency \
  --refresh-hz 120 --record-video --output diagnostics/uvc_front_latency_001 \
  uvc --camera-config calibrations/uvc_front_lrcp_480p_002_camera.json --camera-name front_1
```

两腕使用 `--camera-config calibrations/uvc_rig.json`，并选择 `--camera-name left_wrist` 或 `right_wrist`。
测量读原始接收时间，**不会再减
rig 中已填的 latency**。120 Hz 仅作元数据，不能据此自动扣除一个或半个
刷新周期；有独立实测显示延迟时才传 `--display-latency-ms`。这项测试仍含
相机采样/曝光/首次二维码识别等效应，三路并发负载需另验。

新 LRCP front 必须重新标定到左右 base 的外参，不能使用旧 KD 或 PICO 的结果。
固定好 front 和双臂后，左右臂各采一份外参会话。以下命令会连接真机，
需要现场操作，离线测试不会自动执行它们：

```bash
python -m egomimic.scripts.calibrate_piper_pico collect \
  --arm left --camera-rig calibrations/uvc_front_lrcp_480p_002_camera.json --camera-name front_1 \
  --output calibrations/uvc_left_base_001
python -m egomimic.scripts.calibrate_piper_pico collect \
  --arm right --camera-rig calibrations/uvc_front_lrcp_480p_002_camera.json --camera-name front_1 \
  --output calibrations/uvc_right_base_001
python -m egomimic.scripts.calibrate_piper_pico solve --session calibrations/uvc_left_base_001
python -m egomimic.scripts.calibrate_piper_pico solve --session calibrations/uvc_right_base_001
python -m egomimic.scripts.calibrate_piper_pico export \
  --left-result calibrations/uvc_left_base_001/calibration_result.json \
  --right-result calibrations/uvc_right_base_001/calibration_result.json \
  --camera-rig calibrations/uvc_front_lrcp_480p_002_camera.json --calibration-id lrcp-front-480p-001 \
  --output calibrations/uvc_rig_001_calibration.json
```

仍用原 ChArUco 板、原求解器及留出验证。导出核对两份结果的机械臂角色、
相机、K/D、原图尺寸和采集设置；保留原始内参和 `T_base_camera`。
标定板检测阈值可按原 CLI 显式设置，但不能以放宽阈值代替检查模糊或板面变形。
支架或机械臂位置变化后重新标定，并给新的批次分配标定文件。

## 3. 正式采集、转换和推理图像一致性

三相机采集使用已更新的 `uvc_rig.json`，启动时为三台应用手动曝光 20 ms、亮度 48；运行前核对当前机位画面。
数据转换需要当前 LRCP front 到左右 base 的新外参；单 front 配置继续用于独立标定。

collect 默认显示独立的三相机小窗口：左腕 / front / 右腕，每路 320×240，整体约 960×304，预览 15 fps。
窗口直接从现有 shm 复制最新帧；缩放、拼接和显示都在预览进程中，不改变保存图像或控制循环。
`--no-preview` 关闭窗口，`--preview-fps 10` 调低显示刷新率。窗口 Q/Esc/关闭按钮只退出预览，采集继续。
PICO 按键照常工作；键盘采集快捷键需要终端焦点。无桌面时自动跳过窗口。
窗口内 `camera` 帧率按 shm 发布计数计算，`View` 是显示刷新率；`received` 是主机收帧后的经过时间，不是相机延迟。
超过 0.5 秒没有更新显示 `STALE`。独立进程隔离窗口等待，但机器资源仍共享，不能仅凭显示刷新率判断录制和控制频率。

2026-11-06 做了关闭 / 开启 / 再关闭预览的三轮对照，每轮 20 秒，三台真实相机同时录像。
开启预览时三路分别为 30.055 / 30.053 / 30.051 fps，采集循环 30.001 Hz；三轮均无超过 100 ms 的相机帧间隔，视频帧数与时间戳数一致。
测试运行正式 `get_obs` / `run_teleop`，机械臂和 PICO 输入用模拟数据代替，未发送 CAN 命令；不代表完整真机遥操负载已验证。
结果见 [对照摘要](../../../diagnostics/collect_preview_20261106/ab_001/summary.json)。

```bash
python -m egomimic.scripts.collect_piper_pico \
  --left-can can0 --right-can can1 --frequency 30 \
  --camera-config calibrations/uvc_rig.json --output piper_uvc_capture_001

python -m egomimic.scripts.piper_pico_to_zarr \
  --input piper_uvc_capture_001 --output-dir converted_real \
  --calibration calibrations/uvc_rig_001_calibration.json \
  --width 640 --height 480 --undistort-alpha 0 \
  --max-frame-age 0.2 --task-name teleop
```

转换使用录制时保存的内参快照，并与外参文件中的原图 K/D 核对。
有测量畸变的视角执行：原图 → 用保存的 K_new 去畸变 → 显式 resize → JPEG。
默认 alpha=0，**不裁 ROI、不旋转光学坐标系**。训练图像的内参相应更新为
缩放后的 K_new，D 为零；原始 K/D、输入尺寸和处理参数另存，带校验摘要。
没有内参的腕图保留原来的 resize 路径，不伪造它们的畸变参数。

位姿、命令保持各自 robot base 坐标系，时间戳和动作行选择不变；训练加载时
仍按该数据集外参转到 front_1 光学系。去畸变不改变该光学坐标系的朝向。

`PiperObservationAdapter` 对实时 raw 图像调用同一处理模块。已转换 Zarr 的
`RecordedObservationProvider` 明确标记为 processed，避免再去畸变。
新数据处理尺寸与模型输入不一致、K/D/焦距不一致、处理元数据被改动都会报错。
旧数据不带畸变标定时继续兼容旧图像路径；但更早的数据若缺少控制决策或选图时间戳，
仍按原转换器规则报错，不自动补造时间戳。本项只完善图像适配接口，**不新增实机
CAN 推理执行器**；现有无 CAN 策略回放仍走其原入口。

## 4. 离线复验

```bash
python -m egomimic.scripts.verify_uvc_pipeline \
  --output diagnostics/uvc_pipeline_acceptance_001
```

该脚本生成明确标记的合成三流录像和外参，核对真实分辨率的图像转换、投影、
JPEG 和时间/动作数据，保存 JSON 与比较图片。可加 `--real-input` 检查既有
采集目录的兼容性，输出放到新目录，原始文件不改动。合成验收不是硬件标定。

分阶段测试在 `tests/unit/`：

| 阶段 | 新增测试文件 |
|---|---|
| 1 | `test_uvc_camera_config.py`、`test_uvc_controls.py`、`test_uvc_worker_settings.py`、`test_camera_rig_diagnostic.py` |
| 2 | `test_uvc_extrinsic_config.py`、`test_uvc_latency_config.py` |
| 3 | `test_uvc_image_geometry.py`、`test_uvc_conversion_geometry.py`、`test_uvc_policy_images.py` |
| 4 | `test_uvc_pipeline_acceptance.py`，并回归原转换、加载、策略和训练 step 测试 |

所有输出目录要求为新目录。硬件验收和软件验收分别记录，未连接相机时不能
把离线测试通过解释成三台真实相机已稳定运行。

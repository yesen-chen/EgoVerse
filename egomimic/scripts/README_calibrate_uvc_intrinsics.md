# UVC 内参和畸变实验

独立实验入口：`python -m egomimic.scripts.calibrate_uvc_intrinsics`。
只打开指定 UVC，相机未连接也可以运行离线 `solve`。不需要 PICO、PCService 或 CAN。
求解不会自动修改相机配置；将原始 `calibration.json` 作为 rig 的 `calibration_file` 后，采集与转换会使用该结果。

当前前置 LRCP 已完成 640×480@30 MJPG 内参标定：`calibrations/uvc_front_lrcp_480p_002_result/calibration.json`。
49 张训练图 RMS 0.217 px、11 张留出图 RMS 0.225 px；没有软件对焦控制，保持镜头固定。
当前单 front 配置 `calibrations/uvc_front_lrcp_480p_002_camera.json` 已引用这份原始内参，front 正式设置为亮度 48、手动曝光 20 ms。
三相机 `calibrations/uvc_rig.json` 三路也统一为上述设置；启动时应用并核对 `brightness=48`、`auto_exposure=1`、`exposure_time_absolute=200`。
接口已确认 front `5.3.2`、左腕 `5.3.3`、右腕 `5.3.4`。原始内参会话采用亮度 32、自动曝光模式 3，历史记录保持原样；仅调整亮度和曝光不改变这份几何内参的适用分辨率、焦点及镜头条件。
旧 KD/1080p 标定仅适用于历史相机和数据，不应用到这台替换相机。

## 1. 固定相机模式和标定板

- 继续使用现有 **6×6 格 ChArUco 板**：方格 **19 mm**、标记约 **13.3 mm**、字典 `DICT_5X5_100`、`legacy_pattern=false`。测量的是单格边长，不是内部角点数量。板应平整；如果重新打印，核对实际尺寸和标记/方格比例。
- 使用与实际采集一致的分辨率、焦点和变焦。定焦镜头固定镜头环；自动对焦相机先在工作距离对好焦，再关闭自动对焦。标定期间不要拧镜头或改焦点。
- 本实验只接受原始输出：不翻转、不裁剪、不缩放保存的图片。窗口缩放不影响保存的原图。
- 一个分辨率/镜头设置对应一个采集会话。不同相机、焦点、变焦或驱动裁剪模式不能直接复用同一份内参。

先关闭其他占用相机的程序，在桌面终端检查设备及模式：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --list-ctrls
```

当前三台 LRCP 序列号重复，`by-id` 名称存在冲突；front 用 `/dev/v4l/by-path/...` 固定 USB 位置。
保持同一插口可避开 `/dev/videoN` 编号变化。换口时同步更新配置中的设备路径和预期 `ID_PATH`；
同一台相机只换 USB 口不需要重做内参，原始标定保留当时接口记录。
下面按当前 front 的 640×480@30 MJPG 示范重新采集，输出用新目录，已有标定无需重复覆盖。
驱动未采用请求模式时脚本会报错，不会自动缩放图片。

## 2. 采集

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
v4l2-ctl -d /dev/v4l/by-path/pci-0000:0f:00.0-usb-0:5.3.2:1.0-video-index0 \
  --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/v4l/by-path/pci-0000:0f:00.0-usb-0:5.3.2:1.0-video-index0 \
  --set-ctrl=brightness=48,exposure_time_absolute=200
python -m egomimic.scripts.calibrate_uvc_intrinsics capture \
  --device /dev/v4l/by-path/pci-0000:0f:00.0-usb-0:5.3.2:1.0-video-index0 \
  --name front_1 \
  --width 640 --height 480 --fps 30 --fourcc MJPG \
  --focus-note 'No software focus control; keep lens fixed; brightness=48; manual exposure=20ms' \
  --output calibrations/uvc_front_lrcp_480p_003
```

若存在自动对焦控制项且已对好焦，在命令中加 `--lock-focus`；脚本关闭自动对焦并核对读回值。`--focus N` 则明确设置 `focus_absolute`，**N 是驱动原始单位，不是毫米**。这些设置退出后保留，后续采集应保持一致。若无法读到焦点控制项，会记录为未知，需要手动确认定焦设置。

窗口按键：

- 空格或 **C**：保存当前检测有效的原始 PNG；未检测成功、角点不足或完全重复的图片不保存。
- **V**：切换 `TRAIN` / `VALIDATION`。
- **Q**、Esc 或关闭窗口：释放相机并结束。Ctrl+C 也保留已保存的图片。

建议 **25–35 张训练图、8–10 张独立验证图**。默认求解至少需 15 张有效训练图、5 张验证图。先采训练图，再切到验证，改变板的位置和朝向重新采集。不要把同一张图片复制到两组；也不要连续保存几乎不动的板。

让板覆盖中心、四边、四角，改变距离并向多个方向倾斜；每次停稳、清晰后保存。不要只有居中的正面图。验证图也要覆盖这些位置。尽可能让板占据较多画面，同时保持足够多的可见角点。

会话保存原图、板规格、实际模式、设备身份、对焦/变焦读回值。若需要其他板，可通过 `--board-config board.json` 提供与上述字段对应的 JSON；旧板默认不需要此参数。

## 3. 求解和检查

```bash
python -m egomimic.scripts.calibrate_uvc_intrinsics solve \
  --session calibrations/uvc_front_lrcp_480p_003 \
  --output calibrations/uvc_front_lrcp_480p_003_result
```

求解会从 PNG 重新检测角点，只用 `TRAIN` 拟合 OpenCV 针孔 Brown5 模型：

```text
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
D = [k1, k2, p1, p2, k3]
```

`VALIDATION` 固定 K/D，仅估计每张图的标定板位姿，报告重投影误差，不会用验证图重新拟合 K/D。它是留出检验，仍依赖板模型和每图位姿拟合，不是独立几何真值。

结果目录包含：

- `calibration.json`：**原始图像**使用的 K、D；拟合/留出误差、逐图误差、参数不确定性估计、视角/边缘覆盖、输入哈希、质量提示和相机模式。
- `rectified_camera.json`：**按该结果去畸变后**使用的 K_new、D=0。保留原图尺寸，记录有效 ROI，**未裁剪 ROI**。默认 `--alpha 0`；改为 1 时通常保留更多视场和黑边。
- `comparison/*.jpg`：原图与去畸变图对照，最多 10 张。
- `coverage.png`：训练角点绿色、验证角点橙色。

`--max-rms-px` 默认为 1.0 px，用于诊断提示，不是通用精度保证。不能只看总 RMS：同时检查逐图误差、视角变化、边缘覆盖和图像是否清晰。若边缘仍明显弯曲、参数异常或不同会话差异很大，应检查板规格、平整度、焦点和覆盖后重采；极广角/鱼眼镜头可能需要另选模型。

本阶段始终将 `production_approved` 设为 false，表示实验输出尚未完成实际工作空间验收，不表示求解一定失败。后续需要实际相机验收、相机到左右机械臂的外参，以及训练/推理一致的去畸变流程。不要把 raw K 与去畸变图片配对，也不要把 K_new 与原始畸变图片配对。

输出目录存在时会拒绝覆盖。重新求解时换结果目录；重新采集时换会话目录。

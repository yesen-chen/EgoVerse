# PICO / UVC 观测延迟标定

入口 `egomimic/scripts/calibrate_camera_latency.py` 使用正式采集的相机进程，
没有机械臂、CAN 或手柄控制。PICO 默认请求左目单独输出 1080×810@60；
UVC 默认请求 MJPG 640×480@30。曝光、码率、网络连接方式应与正式采集一致。

本次正式配置采用三台手动曝光 20 ms、亮度 48，并按用户决定沿用已有延迟补偿，未重新测量。
front 继续使用 67.275 ms、两腕继续使用 66 ms；配置保留历史来源，并记录当前曝光设置下延迟未测。
以下测量步骤供以后按需复核，不是本次采集的前置步骤。

## 测试命令（可选复核）

依次运行，不要同时运行多份标定、采集程序或其它相机预览。
当前 front 为替换后的 LRCP，读取 `calibrations/uvc_front_lrcp_480p_002_camera.json` 中的
640×480@30 MJPG、亮度 48、手动曝光 20 ms 和相机身份。测量不会再次扣除 rig 中的延迟。
三相机 `calibrations/uvc_rig.json` 已更新：front `5.3.2`、左腕 `5.3.3`、右腕 `5.3.4`。
每路均明确设置 `brightness=48`、`auto_exposure=1`、`exposure_time_absolute=200`，启动时应用并核对。
左右腕用各自条目测量；这只打开选中的相机，不验证三路并发负载。

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate

# 当前 LRCP front_1
python -m egomimic.scripts.calibrate_camera_latency \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  uvc --camera-config calibrations/uvc_front_lrcp_480p_002_camera.json --camera-name front_1

# 左腕 UVC
python -m egomimic.scripts.calibrate_camera_latency \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  uvc --camera-config calibrations/uvc_rig.json --camera-name left_wrist

# 右腕 UVC
python -m egomimic.scripts.calibrate_camera_latency \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  uvc --camera-config calibrations/uvc_rig.json --camera-name right_wrist
```

PICO 图像的历史/可选模式仍可单独测试；这不使用当前 UVC front。
头显保持 XRoboToolkit 运行并关闭本地 Record/预览，按当前网络修改 IP：

```bash
python -m egomimic.scripts.calibrate_camera_latency \
  --refresh-hz 120 --display-latency-ms 0 --n-frames 240 --record-video \
  pico --headset-ip 192.168.1.9 --pc-ip 192.168.1.8 \
  --mode left --width 1080 --height 810 --fps 60 --bitrate 5000000
```

每次将被测相机对准完整、清晰、无遮挡的二维码，保持约 10 秒，然后在
OpenCV 窗口中按 `c`。程序先关闭相机，再离线分析最近的缓冲；`q`/Esc 退出。
二维码平均约 10 Hz 更新，周期有 ±20% 变化，减少与相机帧周期固定相位锁定。
相机预览最多 30 Hz，不会把预览刷新率当成相机采集率。可重复三次观察结果。

默认缓冲 240 帧：60 fps 对应约 4 秒，30 fps 对应约 8 秒。更久等待不会扩大
这段滚动缓冲。低于默认 `--min-samples 30` 时先检查对焦、二维码可见尺寸和
反光；确需更长统计窗口时再增加 `--n-frames`，图像缓冲内存会随之增加。

`--record-video` 在整个测试期间保存本路 `load_test.mp4` 和时间戳，包含这一路
的编码/磁盘负载。不加该选项则只收流。这两种测试都不等同于三路并发录制；
并行负载、曝光或网络改变可能影响延迟；本次选择沿用旧值，未验证这些变化后的延迟差异。

## 屏幕延迟和结果含义

每个二维码分别记录生成时间 `g`、`imshow` 返回时间 `s` 和首次成功解码帧的
相机进程接收时间 `r`。报告保存三个结果：

- `raw_latency`：`r - g`。
- `latency_before_display_correction`：`r - s`，逐二维码扣除软件生成/提交耗时。
- `latency`：`r - s - display_latency_sec`，只再扣显式输入的剩余屏幕延迟。

`--refresh-hz 120` 只记录 8.33 ms 刷新周期，既不设置显示器刷新率，也不自动扣
8.33 或 4.17 ms。`imshow` 返回不是像素亮起时刻，合成器、扫描和面板响应仍可能
增加延迟。没有独立测量时保持 `--display-latency-ms 0`，报告明确标为未测、未扣。
以后取得与 `imshow` 返回计时点相匹配的实际剩余显示延迟，才填写该值（毫秒）；
不要直接使用显示器广告中的灰阶响应时间。

结果仍包含等待相机采样、曝光和首次成功识别的影响，不是精确的硬件曝光时间标定。
二维码识别本身在测试结束后进行，其 CPU 用时不计入相机接收时间。测量不会自动
修改采集配置，也不会沿用旧的 77 ms / 130 ms。

## 报告和质量检查

每次自动创建新的 `camera_latency_pico_*` 或 `camera_latency_uvc_*` 目录。
可在子命令前指定 `--output 新目录`；已有目录会被拒绝，防止覆盖旧测试。

- `latency_report.json`：三种统计、真实缓冲帧率、参数、每个二维码提交记录、
  有效样本数和剔除原因。`schema=camera_latency_v2`。
- `samples.csv`：逐二维码原始时间、软件耗时、未扣屏幕及扣除后的延迟。
- `latency.png`：未扣屏幕与显式补偿后结果的对比及直方图（有配对时生成）。
- `load_test.mp4` 及时间戳：仅在指定 `--record-video` 时产生。

只匹配本次发出的二维码。缓冲起点之前已提交的二维码会被剔除，因为其真正首次
出现可能已离开缓冲。重复二维码取最早接收时间。负补偿结果会原样保留并报警，
不裁成零。样本不足、负时间或帧率偏离请求值超过 `max(1 fps, 10%)` 时不输出
可直接套用的参数；这些检查通过也不代表已经排除未测显示延迟。

LRCP front 历史三次测试采用亮度 32、自动曝光模式 3，共 226 个有效二维码，合并均值 67.275 ms、中位数 67.107 ms、
P95 85.188 ms。单 front 配置和三相机 rig 的 front 条目都使用 `latency=0.067275` 秒；这是含未测屏幕显示延迟的
单相机录像测试结果，没有重新测量当前 20 ms、亮度 48、三路并发条件下的延迟。旧 KD、PICO 和腕部相机的数值不替代它。
两腕保留 `latency=0.066` 秒；右腕来自此前测量，左腕沿用右腕值、尚未独立测量。
当前三台 `latency_status=unmeasured` 表示新设置下未复测，实际仍按上述非零 `latency` 做补偿，历史来源保留在沿用说明中。

PICO 输出候选参数 `--pico-camera-latency 秒数`；旧 UVC CLI 输出 `--camera-latency 秒数`。
当前推荐的 `--camera-config` 模式由 rig 每路独立填写 `latency` 和 `latency_status`，
无需把 front 的数值覆盖到两腕。测量脚本不会自动修改配置。

旧版 `--view left` 只选读取眼，仍请求双目。新版实际输出由 `--mode` 决定：
`left`/`right` 的 `--width` 是单眼宽；`stereo` 的宽是拼接总宽、默认2160，
再用 `--view left` 或 `right` 选择被测眼。双目模式暂不支持 `--record-video`。

# 推理图像、动作与连续反馈记录

在现有 `python -m egomimic.scripts.infer_piper_policy ...` 命令末尾增加：

```bash
--log --record-images all --record-feedback
```

保存三路连续画面、每次策略请求选中的三张无损图像、请求的当前/历史 state、RTC 前缀与原始预测、执行目标及 SDK 提交回执，以及双臂连续反馈。支持同步、异步和 RTC 推理。包括预热请求；暂停期间仍录相机和反馈，直到会话退出。不依赖 PICO。

图像和连续反馈默认关闭；原有 `--log` 默认开启，并新增请求数值记录。此功能不修改训练数据划分或模型训练。

## 配置

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--record-images off\|requests\|all` | `off` | 关闭图像；仅请求快照；全部连续帧并保存请求快照 |
| `--record-feedback` / `--no-record-feedback` | 关闭 | 双臂 SHM 连续反馈，当前发布频率约 100 Hz |
| `--record-video-format mp4\|lossless` | `mp4` | 连续画面采用分段 H.264 MP4 或每帧无损 PNG |
| `--record-segment-seconds` | `60` | MP4 分段时长，按接收时间划分 |
| `--record-queue-capacity` | `64` | 每个后台写入队列/请求队列容量 |
| `--record-cache-frames` | `90` | 每路后台原始帧缓存数量，用于匹配请求 |
| `--output-dir` | 自动生成 | 本次会话目录；必须是新目录 |

仅保存请求图像与连续反馈：

```bash
--log --record-images requests --record-feedback
```

所有帧均无损保存：

```bash
--log --record-images all --record-video-format lossless --record-feedback
```

关闭全部逐条记录：

```bash
--no-log --record-images off --no-record-feedback
```

图像/反馈记录要求 `--log`，避免图像缺少请求、动作及标定关联。冲突组合在设备打开之前报错。`--offline-eval` 不接受这些实时记录开关。

## 目录与关联

```text
本次 output-dir/
  arguments.json                 启动参数
  manifest.json                  模型、任务、标定、预处理、动作时间定义、源码校验值
  requests.jsonl                 每次预测 started/completed/failed，含预热
  commands.jsonl                 执行目标与双臂 SDK 提交回执（只观察模式明确标记）
  cycles.jsonl                   执行计划、RTC 接管、平滑与阶段结果
  events.jsonl                   开始、暂停、home、hold、失败等事件
  summary.json                   会话状态与各记录通道完整性
  recording/
    summary.json                 图像/反馈数量、缺失、溢出和写入错误
    cameras/
      front_1/                   left_wrist/、right_wrist/ 同结构
        requests.jsonl           request_id → frame_id → PNG，含像素 SHA256
        requests/frame_*.png     精确请求快照，同一帧重复请求复用文件
        frames.jsonl             连续帧编号、源时间、接收时间、文件和视频帧位置
        frames/segment_*.mp4     默认连续画面
        frames/*.png             lossless 模式替代 MP4
    feedback/
      left.jsonl                 逐条原始共享内存反馈
      right.jsonl
```

根目录 `requests.jsonl` 的 `request_id` 是会话内预测唯一编号，`policy_phase=warmup` 表示预热，`phase` 是 started/completed/failed。`cycles.jsonl` 的 `audit_request_id` 关联这个编号，RTC 自己的 `request_id` 单独保留，二者不要混用。相机请求索引使用相同编号（读取时统一转字符串）。请求记录包含 `image_refs`、接收时间、当前/历史 state 及偏移、RTC 前缀和完整预测。`completed` 仅表示模型返回；暂停、超时或接管拒绝后的预测可能完全没有执行。

PNG 保存该请求选中的**预处理前原始 RGB**。定量回放按 `manifest.json` 的 `runtime_metadata` 和对应预处理代码生成模型输入；`recording_contract.pipeline_sha256` 可核验代码版本。MP4 有损且按名义 FPS 播放，不能代替无损请求图像；精确时间使用逐帧索引，掉帧时不能用 `video_frame / FPS` 当采集时间。

后台必须匹配同一路的 `frame_id` 和源时间戳。找不到精确帧记录 `frame_unavailable`，时间不一致记录 `timestamp_mismatch`，不回退到最新帧。实际推理取帧及后台取帧均检查复制预算，拒绝可能被 SHM 绕圈覆盖的图像。

`commands.jsonl` 保存 RTC 混合/平滑后的目标，`requests.jsonl` 保存执行侧夹爪裁剪前的预测，`cycles.jsonl` 保存接入的计划。指令失败或双臂只有一侧确认等情况另记 `command_failed`，不能当成双臂成功执行。SDK 回执仅表示主机调用返回，不代表机器人到达目标。用回执的 `command_id`、`host_send_timestamp` 和反馈的 `last_command_id`、`last_command_send_timestamp` 对照反馈曲线。

原始反馈 `ee_pose` 为各臂 base 下 **xyz 米 + xyz Euler/RPY 弧度**；请求 `base_state` 为每臂 **xyz 米 + 旋转向量弧度 + 夹爪米**，左右拼成 14 维。反馈另存关节、夹爪、主机读取时间及各 CAN 源时间。`feedback_id` 表示一次 SHM 发布；源时间未变时可能仍为同一份 CAN 缓存，不代表新硬件测量。NaN 字段在 JSON 中保存为 null。

## 性能与故障边界

每路相机独立的低优先级进程只读现有 SHM，不重新打开相机。后台编码线程单线程，图像复制、编码、哈希及文件操作都不在策略/动作进程中进行。反馈有独立进程；请求数值日志与控制日志也有独立队列和进程。策略侧只提交小型帧引用及数值快照，不通过父进程队列发送图像。

缓存或队列满时丢弃记录，不等待空位；记录失败不会仅因日志问题暂停机械臂。持续写入不足、进程故障或异常退出会设置 `incomplete`，摘要包含缺失范围、队列丢弃、请求缺失、读超时和未排空状态。范围列表本身有上限，超出后仍累计数量。丢失可能到退出摘要才能完整统计。

启动和退出会等待后台进程；退出前先保持机械臂，再停止记录，通常最多等待约 5 秒排空，强制回收另有短暂等待。最终摘要在控制结束后写盘，文件系统整体挂死不在此等待上限保证内。`written` 不代表断电持久化；强制终止可能留下不完整视频尾段。

独立进程仍共享 CPU、内存和磁盘，不承诺绝对零开销或资源不足时零丢帧。三路 640×480 RGB、30 FPS 未压缩约 83 MB/s；无损 PNG 的体积依画面变化，CPU/磁盘开销通常更大。缓存与队列可能合计占用数百 MB，容量应按内存调整。增加缓存只能缓冲短暂停顿，不能解决持续写入不足。

## 不连接真机的测试

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_policy_recording.py \
  tests/unit/test_recording_frame_ids.py \
  tests/unit/test_policy_audit.py \
  tests/unit/test_live_policy_recording.py \
  tests/unit/test_policy_logging.py
```

用模拟三路 640×480、30 Hz 图像和双臂 100 Hz 反馈，比较记录关闭/开启：

```bash
python -m egomimic.scripts.check_policy_recording \
  --output-dir /tmp/piper_recording_benchmark_001 \
  --seconds 10 --width 640 --height 480
```

输出目录必须不存在。`benchmark.json` 含两组发布 FPS、模拟动作间隔 P50/P95/P99、请求审计耗时（含模拟预测及后台提交）和记录完整性。开启组同时启用图像、反馈、请求数值和控制日志。测试不访问相机、CAN、GPU 或模型，不能替代带实际模型的真机性能验收。

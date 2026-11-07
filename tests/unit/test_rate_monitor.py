"""Camera production and source health must not be inferred from poll clocks."""
import numpy as np
import pytest

from egomimic.robot.piper_pico.rate_monitor import RateMonitor, observation_k


def _pose(index):
    return {
        "receive_timestamp": 1000 + index / 30,
        "source_timestamp_ns": 1 + round(index * 1e9 / 30),
        "source_age_s": 0.0,
        "source_stall_count": 0,
        "source_max_gap_s": 1 / 30,
        "left_pos": np.array([0.1, 0.2, 0.3]),
        "right_pos": np.array([-0.1, 0.2, 0.3]),
        "left_quat_xyzw": np.array([0., 0., 0., 1.]),
        "right_quat_xyzw": np.array([0., 0., 0., 1.]),
    }


def _camera(index, now=None, stamp=None):
    now = index / 30 if now is None else now
    stamp = now if stamp is None else stamp
    data = {"camera_copy_s": 0.002}
    for side in ("left", "right"):
        data.update({
            f"camera_{side}_count": index + 1,
            f"camera_{side}_count_mono": now,
            f"camera_{side}_receive_timestamp": stamp,
            f"camera_{side}_age_s": max(0.0, now - stamp),
        })
    return data


def _mono_camera(side, index, now=None, stamp=None):
    return {key: value for key, value in _camera(index, now, stamp).items()
            if key == "camera_copy_s" or key.startswith(f"camera_{side}_")}


def test_observation_history_matches_the_collector():
    assert observation_k(2, 30.0) == 6


def test_steady_30_hz_passes():
    monitor = RateMonitor()
    for index in range(120):
        monitor.observe(_pose(index), _camera(index), now_mono=index / 30)
    assert monitor.camera_hz("left") == pytest.approx(30)
    assert monitor.failures(cmd_jumps=0) == []


@pytest.mark.parametrize("side,label", [("left", "左眼"), ("right", "右眼")])
def test_monocular_camera_requires_only_the_selected_eye(side, label):
    monitor = RateMonitor(camera_mode=side)
    for index in range(120):
        monitor.observe(_pose(index), _mono_camera(side, index), now_mono=index / 30)
    assert monitor.failures(0) == []
    assert monitor.camera_hz(side) == pytest.approx(30)
    assert monitor.camera_misses == monitor.camera_diagnostics_missing == 0
    assert monitor.read_phase_skew_ms == []
    assert f"解码发布 {label} 30.0 fps" in monitor.line()
    assert f"控制拍新帧 {label} 30.0 Hz" in monitor.line()
    report = monitor.report(0)
    assert f"控制拍新帧累计 {label} 30.0 Hz" in report
    assert "相位差：不适用（单目模式）" in report
    assert "相位差95% 0.0" not in report
    assert "解码发布 左/右" not in report


@pytest.mark.parametrize("side,label", [("left", "左眼"), ("right", "右眼")])
def test_monocular_missing_selected_eye_is_reported(side, label):
    other = "right" if side == "left" else "left"
    monitor = RateMonitor(camera_mode=side)
    for index in range(60):
        # A healthy inactive eye must never stand in for the requested one.
        monitor.observe(_pose(index), _mono_camera(other, index), now_mono=index / 30)
    problems = monitor.failures(0)
    assert any(f"{label}累计解码发布帧率未知" in item for item in problems)
    assert any("缺图 60" in item for item in problems)


@pytest.mark.parametrize("side,label", [("left", "左眼"), ("right", "右眼")])
def test_monocular_selected_eye_stall_is_reported(side, label):
    monitor = RateMonitor(camera_mode=side, window_s=2)
    for index in range(180):
        latest = min(index, 119)
        data = _mono_camera(side, latest, now=index / 30, stamp=latest / 30)
        monitor.observe(_pose(index), data, now_mono=index / 30)
    problems = monitor.failures(0)
    assert any(f"{label}近2s解码发布" in item for item in problems)
    assert any(f"{label}最新图像最大接收年龄" in item for item in problems)


@pytest.mark.parametrize("mode", ["stereo", "left", "right"])
def test_camera_disabled_omits_camera_diagnostics_in_all_modes(mode):
    monitor = RateMonitor(camera_enabled=False, camera_mode=mode)
    for index in range(60):
        monitor.observe(_pose(index), now_mono=index / 30)
    assert monitor.failures(0) == []
    report = monitor.report(0)
    for text in ("解码发布", "控制拍新帧", "读图", "相机", "相位差"):
        assert text not in report


@pytest.mark.parametrize("mode", ["left", "right"])
def test_monocular_camera_still_checks_both_controllers(mode):
    monitor = RateMonitor(camera_mode=mode)
    for index in range(60):
        sample = _pose(index)
        sample["left_pos"] = sample["right_pos"] = np.zeros(3)
        monitor.observe(sample, _mono_camera(mode, index), now_mono=index / 30)
    assert monitor.origin_ticks == {"left": 60, "right": 60, "both": 60}
    assert any("origin pose：左 60 拍，右 60 拍，同时 60 拍" in item
               for item in monitor.failures(0))


@pytest.mark.parametrize("mode", ["mono", "both", "", None])
def test_invalid_camera_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="camera_mode"):
        RateMonitor(camera_mode=mode)


def test_default_stereo_report_retains_both_eye_rates():
    monitor = RateMonitor()
    for index in range(60):
        monitor.observe(_pose(index), _camera(index), now_mono=index / 30)
    assert "解码发布 左/右 30.0/30.0 fps" in monitor.line()
    assert "控制拍新帧累计 左/右 30.0/30.0 Hz" in monitor.report(0)
    assert "左右最新帧读取相位差95% 0.0 ms" in monitor.report(0)


def test_asynchronous_latest_sampling_does_not_underreport_producer_fps():
    # Realistic +/-1ms arrival jitter is sufficient for the old latest-only
    # timestamp counter to report ~27.4Hz for a 30fps producer.
    arrivals = np.arange(3000) / 30 + np.random.default_rng(11).uniform(-0.001, 0.001, 3000)
    polls = np.arange(3000) / 30 + 0.0008
    monitor = RateMonitor()
    for index, now in enumerate(polls):
        latest = int(np.searchsorted(arrivals, now, side="right") - 1)
        if latest >= 0:
            monitor.observe(_pose(index), _camera(latest, now, arrivals[latest]), now_mono=now)
    assert monitor.camera_hz("left") == pytest.approx(30, abs=0.02)
    assert 27 < monitor.camera_hz("left", consumer=True) < 28
    assert monitor.failures(0) == []


def test_faster_control_recovers_unique_frames_for_small_arrival_jitter():
    # Polling at 60Hz covers every 30fps arrival in this bounded-jitter case.
    # This is not a guarantee for network bursts or a stalled consumer.
    arrivals = np.arange(1200) / 30 + np.random.default_rng(11).uniform(-0.001, 0.001, 1200)
    monitor = RateMonitor(control_hz=60, camera_fps=30)
    for index, now in enumerate(np.arange(2400) / 60 + 0.0008):
        latest = int(np.searchsorted(arrivals, now, side="right") - 1)
        if latest >= 0:
            monitor.observe(_pose(index), _camera(latest, now, arrivals[latest]), now_mono=now)
    assert monitor.camera_hz("left") == pytest.approx(30, abs=0.05)
    assert monitor.camera_hz("left", consumer=True) == pytest.approx(30, abs=0.05)
    assert monitor.failures(0) == []


def test_reading_different_stereo_publications_is_not_exposure_sync_failure():
    monitor = RateMonitor()
    for index in range(120):
        data = _camera(index)
        data["camera_left_receive_timestamp"] -= 1 / 30
        data["camera_left_age_s"] = 1 / 30
        monitor.observe(_pose(index), data, now_mono=index / 30)
    assert np.percentile(monitor.read_phase_skew_ms, 95) == pytest.approx(1000 / 30)
    assert monitor.failures(0) == []
    assert "读取相位差" in monitor.report(0)
    assert "不能证明相机曝光同步" in monitor.report(0)


def test_source_stamp_stall_is_detected_even_while_poll_timestamp_advances_without_camera():
    monitor = RateMonitor(camera_enabled=False)
    for index in range(120):
        sample = _pose(index)
        sample["source_timestamp_ns"] = 12345
        # A stopped producer can also leave source_age_s frozen at zero.
        monitor.observe(sample, now_mono=index / 30)
    assert monitor.observed_source_stalls == 1
    assert monitor.source_max_gap_s > 3.9
    assert any("姿态源停更" in problem for problem in monitor.failures(0))
    assert "解码发布" not in monitor.report(0)


def test_origin_and_missing_source_diagnostics_do_not_pass_without_camera():
    monitor = RateMonitor(camera_enabled=False)
    for index in range(60):
        sample = _pose(index)
        del sample["source_timestamp_ns"]
        sample["left_pos"] = sample["right_pos"] = np.zeros(3)
        monitor.observe(sample, now_mono=index / 30)
    assert monitor.origin_ticks == {"left": 60, "right": 60, "both": 60}
    failures = monitor.failures(0)
    assert any("origin pose" in item for item in failures)
    assert any("姿态诊断不完整" in item for item in failures)


def test_recent_camera_stall_is_visible_despite_good_cumulative_fps():
    monitor = RateMonitor(window_s=2)
    for index in range(600):
        # Healthy for 18 seconds, then stop producing for the final 2 seconds.
        latest = min(index, 539)
        monitor.observe(_pose(index), _camera(latest, index / 30, latest / 30), now_mono=index / 30)
    assert monitor.camera_hz("left") >= 26.9
    assert monitor.camera_hz("left", recent=True) < 1
    failures = monitor.failures(0)
    assert any("近2s解码发布" in item for item in failures)
    assert any("接收年龄" in item for item in failures)


def test_requested_fps_controls_camera_acceptance_threshold():
    monitor = RateMonitor(camera_fps=15)
    for index in range(120):
        latest = index // 2
        monitor.observe(_pose(index), _camera(latest, index / 30, latest / 15), now_mono=index / 30)
    assert monitor.camera_hz("left") == pytest.approx(15, abs=0.2)
    assert monitor.failures(0) == []
    assert any("0.1 m" in item for item in monitor.failures(cmd_jumps=1))


def test_missing_camera_counters_fail_instead_of_falling_back_to_latest_frame_rate():
    monitor = RateMonitor()
    for index in range(60):
        data = _camera(index)
        del data["camera_left_count"]
        del data["camera_right_count"]
        monitor.observe(_pose(index), data, now_mono=index / 30)
    assert any("帧率未知" in item for item in monitor.failures(0))


def test_long_source_sampling_gap_is_reported_without_claiming_a_packet_stall():
    monitor = RateMonitor(camera_enabled=False)
    for index in range(60):
        sample = _pose(index)
        sample["source_max_gap_s"] = 0.2
        monitor.observe(sample, now_mono=index / 30)
    assert monitor.source_stall_count == monitor.observed_source_stalls == 0
    assert any("需区分源停更与采集进程停顿" in item for item in monitor.failures(0))


def test_long_control_pause_cannot_be_hidden_by_catchup_and_good_average_rate():
    monitor = RateMonitor(camera_enabled=False)
    times = np.arange(300, dtype=float) / 30
    # A 200ms pause followed by fast catch-up preserves the total tick rate
    # and p95, while the independently running pose source stays healthy.
    times[60:66] = times[66] - np.arange(6, 0, -1) * 0.001
    for index, now in enumerate(times):
        monitor.observe(_pose(index), now_mono=now)
    assert np.percentile(monitor.periods, 95) == pytest.approx(1 / 30)
    problems = monitor.failures(0)
    assert len(problems) == 1
    assert "控制循环间隔≥100ms 1 次" in problems[0]
    assert "本次观测在检查阈值内" not in monitor.report(0)


@pytest.mark.parametrize("held,reanchors", [(15, 2), (0, 1)])
def test_protected_controller_jumps_are_reported_even_without_command_jumps(held, reanchors):
    monitor = RateMonitor(camera_enabled=False)
    for index in range(90):
        monitor.observe(_pose(index), now_mono=index / 30)
    assert monitor.failures(0) == []
    problems = monitor.failures(0, held_rows=held, reanchor_rows=reanchors)
    assert len(problems) == 1
    assert "手柄跳变保护" in problems[0]
    report = monitor.report(0, held_rows=held, reanchor_rows=reanchors)
    assert "指令单拍超过 0.1 m：0" in report
    assert "不等于独立异常次数" in report
    assert "本次观测在检查阈值内" not in report

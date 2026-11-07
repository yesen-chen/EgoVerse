"""Episode log, one row per control tick.

A row holds what ``get_obs`` returned at that tick (query time, selected frame
stamps, arm state at the query time) and the command decided after it. The
arm's raw state stream is kept beside the videos; it carries the id and host
send time of every waypoint the arm process actually sent. Images stay in the
mp4 files with a sibling timestamp array.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import zarr


class ObsAccumulator:
    def __init__(self):
        self.data = collections.defaultdict(list)
        self.timestamps = collections.defaultdict(list)

    def put(self, data: dict, timestamps: np.ndarray) -> None:
        stamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        for key, value in data.items():
            samples = np.asarray(value)
            for index, stamp in enumerate(stamps):
                history = self.timestamps[key]
                if len(history) == 0 or history[-1] < stamp:
                    history.append(float(stamp))
                    self.data[key].append(samples[index])

    def stream(self, prefix: str) -> dict[str, np.ndarray]:
        """All keys starting with ``prefix`` with their own timestamps."""
        result = {}
        for key, values in self.data.items():
            if key.startswith(prefix) and values:
                name = key[len(prefix):]
                result[name] = np.stack(values, axis=0)
                result[f"{name}__timestamp"] = np.asarray(self.timestamps[key], dtype=np.float64)
        return result


class TickAccumulator:
    def __init__(self):
        self.rows: list[dict[str, np.ndarray]] = []

    def put(self, row: dict) -> None:
        values = {key: np.asarray(value).copy() for key, value in row.items()}
        if self.rows and set(values) != set(self.rows[0]):
            raise ValueError("every tick row must contain the same fields")
        if self.rows and float(values["timestamp"]) <= float(self.rows[-1]["timestamp"]):
            raise ValueError("tick observation times must increase")
        self.rows.append(values)

    def __len__(self) -> int:
        return len(self.rows)


class ReplayBuffer:
    """One zarr group with ``data`` arrays and ``meta/episode_ends``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.root = zarr.open_group(str(self.path), mode="a", zarr_format=3)
        if "data" not in self.root:
            self.root.create_group("data")
        if "meta" not in self.root:
            self.root.create_group("meta")
        if "episode_ends" not in self.root["meta"]:
            self.root["meta"].create_array(
                "episode_ends",
                shape=(0,),
                chunks=(1024,),
                dtype=np.int64,
            )

    @property
    def n_episodes(self) -> int:
        return int(self.root["meta"]["episode_ends"].shape[0])

    @property
    def n_steps(self) -> int:
        if self.n_episodes == 0:
            return 0
        return int(self.root["meta"]["episode_ends"][-1])

    def add_episode(self, episode: dict) -> None:
        lengths = {key: len(value) for key, value in episode.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode arrays have different lengths: {lengths}")
        n_new = next(iter(lengths.values()))
        if n_new <= 0:
            return
        for key, value in episode.items():
            array = np.asarray(value)
            self._append(key, array)
        ends = self.root["meta"]["episode_ends"]
        old = int(ends.shape[0])
        ends.resize((old + 1,))
        ends[old] = self.n_steps_of("timestamp")

    def n_steps_of(self, key: str) -> int:
        if key not in self.root["data"]:
            return 0
        return int(self.root["data"][key].shape[0])

    def drop_episode(self) -> None:
        ends = self.root["meta"]["episode_ends"]
        if ends.shape[0] == 0:
            return
        previous = 0 if ends.shape[0] == 1 else int(ends[-2])
        for key in list(self.root["data"].keys()):
            array = self.root["data"][key]
            array.resize((previous,) + array.shape[1:])
        ends.resize((int(ends.shape[0]) - 1,))

    def _append(self, key: str, value: np.ndarray) -> None:
        data = self.root["data"]
        if key not in data:
            chunks = (min(int(value.shape[0]), 1024),) + value.shape[1:]
            data.create_array(key, shape=value.shape, chunks=chunks, dtype=value.dtype)
            data[key][:] = value
            return
        array = data[key]
        start = int(array.shape[0])
        array.resize((start + value.shape[0],) + array.shape[1:])
        array[start:] = value


def command_send_times(
    command_ids: np.ndarray, sent_ids: np.ndarray, sent_times: np.ndarray
) -> np.ndarray:
    """First reported send time of each command id; NaN if it was never sent.

    A waypoint superseded inside one arm-queue read is never sent.
    """
    first = {}
    for command_id, stamp in zip(np.asarray(sent_ids).reshape(-1), np.asarray(sent_times).reshape(-1)):
        command_id = int(command_id)
        if command_id >= 0 and np.isfinite(stamp) and command_id not in first:
            first[command_id] = float(stamp)
    return np.asarray([first.get(int(i), np.nan) for i in np.asarray(command_ids).reshape(-1)])


def episode_from_ticks(
    ticks: TickAccumulator,
    robot_streams: list[dict[str, np.ndarray]],
) -> dict | None:
    if len(ticks) == 0:
        return None
    episode = {
        key: np.stack([row[key] for row in ticks.rows], axis=0) for key in ticks.rows[0]
    }
    for robot_idx, stream in enumerate(robot_streams):
        if "last_command_id" in stream:
            episode[f"robot{robot_idx}_send_timestamp"] = command_send_times(
                episode["command_id"],
                stream["last_command_id"],
                stream["last_command_send_timestamp"],
            )
        else:
            episode[f"robot{robot_idx}_send_timestamp"] = np.full(len(ticks), np.nan)
    return episode


def save_robot_streams(directory: Path, robot_streams: list[dict[str, np.ndarray]]) -> None:
    directory = Path(directory)
    for robot_idx, stream in enumerate(robot_streams):
        np.savez(directory / f"robot{robot_idx}_state.npz", **stream)

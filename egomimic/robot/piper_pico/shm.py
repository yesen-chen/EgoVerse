"""Single-writer shared-memory ring and command queue.

The layout matches the UMI ring used by the bimanual collector: one process
writes, any process may read the latest samples, and a separate queue carries
numeric commands. Counters are 64-bit values in shared memory. Each structure
has one writer.
"""

from __future__ import annotations

import numbers
import time
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from queue import Empty, Full
from typing import Dict, List, Tuple, Union

import numpy as np


@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype


class SharedNDArray:
    def __init__(self, shm, shape: Tuple[int, ...], dtype: np.dtype):
        self.shm = shm
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @classmethod
    def create_from_shape(cls, mem_mgr: SharedMemoryManager, shape, dtype):
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        shm = mem_mgr.SharedMemory(size=max(int(nbytes), 1))
        array = cls(shm, shape, dtype)
        array.get()[...] = 0
        return array

    def get(self) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)


class SharedCounter:
    def __init__(self, mem_mgr: SharedMemoryManager):
        self._array = SharedNDArray.create_from_shape(mem_mgr, shape=(1,), dtype=np.uint64)

    def load(self) -> int:
        return int(self._array.get()[0])

    def store(self, value: int) -> None:
        self._array.get()[0] = np.uint64(value)

    def add(self, value: int) -> None:
        current = self.load()
        self.store(current + int(value))


def join_process(process, timeout: float) -> None:
    """Join a process only after it has been started."""
    if process.pid is None:
        return
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)


def _specs_from_examples(examples: Dict[str, Union[np.ndarray, numbers.Number]]) -> List[ArraySpec]:
    specs = []
    for key, value in examples.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.dtype("O"):
                raise TypeError(f"{key} has object dtype")
            shape = tuple(value.shape)
            dtype = value.dtype
        elif isinstance(value, numbers.Number):
            shape = tuple()
            dtype = np.dtype(type(value))
        else:
            raise TypeError(f"Unsupported type for {key}: {type(value)}")
        specs.append(ArraySpec(name=key, shape=shape, dtype=dtype))
    return specs


class SharedMemoryRingBuffer:
    """Lock-free FILO ring of dicts of numpy arrays. One writer."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        array_specs: List[ArraySpec],
        get_max_k: int,
        get_time_budget: float,
        put_desired_frequency: float,
        safety_margin: float = 1.5,
    ):
        self.buffer_size = int(
            np.ceil(put_desired_frequency * get_time_budget * safety_margin)
        ) + int(get_max_k)
        self.array_specs = list(array_specs)
        self.counter = SharedCounter(shm_manager)
        self.shared_arrays = {
            spec.name: SharedNDArray.create_from_shape(
                shm_manager,
                shape=(self.buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype,
            )
            for spec in array_specs
        }
        self.timestamp_array = SharedNDArray.create_from_shape(
            shm_manager, shape=(self.buffer_size,), dtype=np.float64
        )
        self.timestamp_array.get()[:] = -np.inf
        self.get_time_budget = float(get_time_budget)
        self.get_max_k = int(get_max_k)
        self.put_desired_frequency = float(put_desired_frequency)

    @classmethod
    def create_from_examples(
        cls,
        shm_manager: SharedMemoryManager,
        examples: Dict[str, Union[np.ndarray, numbers.Number]],
        get_max_k: int = 32,
        get_time_budget: float = 0.01,
        put_desired_frequency: float = 60,
    ) -> "SharedMemoryRingBuffer":
        return cls(
            shm_manager=shm_manager,
            array_specs=_specs_from_examples(examples),
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency,
        )

    @property
    def count(self) -> int:
        return self.counter.load()

    def clear(self) -> None:
        self.counter.store(0)

    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]], wait: bool = True) -> None:
        count = self.counter.load()
        next_idx = count % self.buffer_size
        lookahead = (next_idx + self.get_max_k - 1) % self.buffer_size
        old_timestamp = float(self.timestamp_array.get()[lookahead])
        now = time.monotonic()
        if (now - old_timestamp) < self.get_time_budget:
            remaining = self.get_time_budget - (now - old_timestamp)
            if wait:
                time.sleep(remaining)
            else:
                raise TimeoutError("ring put exceeded its time budget")
        for key, value in data.items():
            slot = self.shared_arrays[key].get()
            slot[next_idx] = value
        self.timestamp_array.get()[next_idx] = time.monotonic()
        self.counter.add(1)

    def _allocate_empty(self, k: int | None = None) -> Dict[str, np.ndarray]:
        result = {}
        for spec in self.array_specs:
            shape = spec.shape if k is None else (k,) + tuple(spec.shape)
            result[spec.name] = np.empty(shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> Dict[str, np.ndarray]:
        if self.count <= 0:
            raise Empty()
        if out is None:
            out = self._allocate_empty()
        curr_idx = (self.counter.load() - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            np.copyto(out[key], value.get()[curr_idx])
        return out

    def get_last_k(self, k: int, out=None) -> Dict[str, np.ndarray]:
        if k > self.get_max_k:
            raise ValueError(f"k={k} exceeds get_max_k={self.get_max_k}")
        count = self.counter.load()
        if k > count:
            raise ValueError(f"k={k} exceeds count={count}")
        if out is None:
            out = self._allocate_empty(k)
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            source = value.get()
            target = out[key]
            end = curr_idx + 1
            start = max(0, end - k)
            target_end = k
            target_start = target_end - (end - start)
            target[target_start:target_end] = source[start:end]
            remainder = k - (end - start)
            if remainder > 0:
                target[0:remainder] = source[self.buffer_size - remainder : self.buffer_size]
        return out

    def get_all(self) -> Dict[str, np.ndarray]:
        k = min(self.count, self.get_max_k)
        if k <= 0:
            raise Empty()
        return self.get_last_k(k=k)


class SharedMemoryQueue:
    """Lock-free FIFO of dicts of numpy arrays. One writer, one reader."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        array_specs: List[ArraySpec],
        buffer_size: int,
    ):
        self.buffer_size = int(buffer_size)
        self.array_specs = list(array_specs)
        self.write_counter = SharedCounter(shm_manager)
        self.read_counter = SharedCounter(shm_manager)
        self.shared_arrays = {
            spec.name: SharedNDArray.create_from_shape(
                shm_manager,
                shape=(self.buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype,
            )
            for spec in array_specs
        }

    @classmethod
    def create_from_examples(
        cls,
        shm_manager: SharedMemoryManager,
        examples: Dict[str, Union[np.ndarray, numbers.Number]],
        buffer_size: int,
    ) -> "SharedMemoryQueue":
        return cls(
            shm_manager=shm_manager,
            array_specs=_specs_from_examples(examples),
            buffer_size=buffer_size,
        )

    def qsize(self) -> int:
        return self.write_counter.load() - self.read_counter.load()

    def empty(self) -> bool:
        return self.qsize() <= 0

    def clear(self) -> None:
        self.read_counter.store(self.write_counter.load())

    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]]) -> None:
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        if write_count - read_count >= self.buffer_size:
            raise Full()
        next_idx = write_count % self.buffer_size
        for key, value in data.items():
            self.shared_arrays[key].get()[next_idx] = value
        self.write_counter.add(1)

    def _allocate_empty(self, k: int | None = None) -> Dict[str, np.ndarray]:
        result = {}
        for spec in self.array_specs:
            shape = spec.shape if k is None else (k,) + tuple(spec.shape)
            result[spec.name] = np.empty(shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        if write_count - read_count <= 0:
            raise Empty()
        if out is None:
            out = self._allocate_empty()
        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            np.copyto(out[key], value.get()[next_idx])
        self.read_counter.add(1)
        return out

    def get_k(self, k: int, out=None) -> Dict[str, np.ndarray]:
        n_data = self.qsize()
        if n_data <= 0:
            raise Empty()
        if k > n_data:
            raise ValueError(f"k={k} exceeds queued {n_data}")
        out = self._get_k_impl(k, self.read_counter.load(), out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> Dict[str, np.ndarray]:
        n_data = self.qsize()
        if n_data <= 0:
            raise Empty()
        out = self._get_k_impl(n_data, self.read_counter.load(), out=out)
        self.read_counter.add(n_data)
        return out

    def _get_k_impl(self, k: int, read_count: int, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty(k)
        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            source = value.get()
            target = out[key]
            start = curr_idx
            end = min(start + k, self.buffer_size)
            copied = end - start
            target[0:copied] = source[start:end]
            remainder = k - copied
            if remainder > 0:
                target[copied:k] = source[0:remainder]
        return out

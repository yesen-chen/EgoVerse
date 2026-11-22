import copy
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import torch
from fsspec.implementations.local import LocalFileSystem
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

import egomimic.utils.hydra_resolvers  # noqa: F401  -- registers OmegaConf resolvers
from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_model import ModelWrapper, checkpoint_values_equal
from egomimic.rldb.action_timing import (
    parse_action_timing,
    validate_dataset_timing,
    validate_model_timing,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.resolve_memo import resolve_once
from egomimic.rldb.zarr.local_recordings import (
    local_data_contracts,
    validate_recording_split,
)
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, PinError
from egomimic.utils.env import load_env
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)

log = RankedLogger(__name__, rank_zero_only=True)


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "norm_stats" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.norm_stats = None
    timing = _action_timing_from_config(cfg)
    contracts = local_data_contracts(cfg)
    if not timing and not contracts:
        return OmegaConf.create({"model": model_cfg})
    # Resolve references while the subtree can still see the complete config.
    # The checkpoint must not depend on a training-only root or config file.
    return OmegaConf.create(
        {
            "model": OmegaConf.to_container(model_cfg, resolve=True),
            "action_timing": {
                name: spec.to_dict(
                    episode_end_padding=(
                        "per_episode_end_reason; see data_contracts."
                        f"{name}.future_window_policy; full-window starts use "
                        "no padding, other starts repeat_last"
                    )
                    if "future_window_policy" in contracts.get(name, {})
                    else None
                )
                for name, spec in timing.items()
            },
            **({"data_contracts": contracts} if contracts else {}),
        }
    )


def _action_timing_from_config(cfg: DictConfig):
    node = OmegaConf.select(cfg, "action_timing", default=None)
    return parse_action_timing(
        OmegaConf.to_container(node, resolve=True) if node is not None else None
    )


def _effective_local_checkpoint_path(cfg):
    """Resolve requeue before reading statistics, not after model construction."""
    path = cfg.get("ckpt_path")
    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        path = os.path.join(
            str(cfg.trainer.default_root_dir), "checkpoints", "last.ckpt"
        )
        with open_dict(cfg):
            cfg.ckpt_path = path
        log.info(
            "Detected local-recipe SLURM requeue — restoring saved statistics and checkpoint"
        )
    return path


def _restore_local_norm_state(checkpoint, current_schema):
    """Reuse saved values after checking today's data interface, without refitting."""
    state = checkpoint.get("hyper_parameters", {}).get("norm_stats_state")
    if not isinstance(state, dict) or not state.get("norm_stats"):
        raise ValueError("Local checkpoint is missing saved norm_stats_state")
    current = current_schema.to_state()
    for key in ("norm_mode", "embodiments", "key_types", "zarr_keys", "shapes"):
        if not checkpoint_values_equal(state.get(key), current.get(key)):
            raise ValueError(f"Checkpoint normalization schema mismatch: {key}")
    saved = MultiDataset.from_state(state)
    required_statistics = {
        "mean",
        "std",
        "min",
        "max",
        "median",
        "quantile_1",
        "quantile_99",
        "quantile_0_01",
        "quantile_99_99",
    }
    if set(saved.norm_stats) != saved.embodiments:
        raise ValueError(
            "Checkpoint normalization statistics have mismatched embodiments"
        )
    for emb in saved.embodiments:
        expected = set(saved.keys_of_type("proprio_keys", emb)) | set(
            saved.keys_of_type("action_keys", emb)
        )
        if set(saved.norm_stats[emb]) != expected:
            raise ValueError(
                f"Checkpoint normalization statistics have missing/extra keys for {emb}"
            )
        for key in expected:
            stats = saved.norm_stats[emb][key]
            if not required_statistics.issubset(stats):
                raise ValueError(
                    f"Checkpoint normalization statistics incomplete: {emb}/{key}"
                )
            for stat, values in stats.items():
                values = np.asarray(values)
                if (
                    values.shape != tuple(saved.shapes[emb][key])
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        f"Invalid checkpoint normalization values: {emb}/{key}/{stat}"
                    )
            if np.any(np.asarray(stats["std"]) < 0):
                raise ValueError(f"Negative checkpoint normalization std: {emb}/{key}")
    return saved


def _log_dataset_frame_counts(train_datasets: dict, valid_datasets: dict) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


def _instantiate_dataset(*args, **kwargs):
    """``hydra.utils.instantiate`` wraps any exception raised by the target in an
    ``InstantiationException``. A bad pin must surface at launch as a bare
    ``PinError`` (not a generic Hydra error the caller has to unwrap), so this
    intercepts only that case and re-raises the original ``PinError`` -- keeping
    the ``InstantiationException`` as its context (``from e``, not ``from None``)
    so the Hydra target/config info isn't lost. Every other target exception is
    re-raised exactly as Hydra raised it."""
    try:
        return hydra.utils.instantiate(*args, **kwargs)
    except hydra.errors.InstantiationException as e:
        if isinstance(e.__cause__, PinError):
            raise e.__cause__ from e
        raise


class MmapCheckpointIO(TorchCheckpointIO):
    """``TorchCheckpointIO`` that memory-maps tensor storages instead of reading them.

    Every DDP rank loads the checkpoint independently and with no rank guard
    (``checkpoint_connector.resume_start``), so a plain read holds N private
    copies of the file in host RAM at once -- N x (weights + optimizer moments),
    and the moments are 2x the weights for AdamW. Mapping the file instead lets
    ranks on a node share page-cache pages, so one physical copy backs them all.

    ``pl_load`` does not forward ``mmap``, hence the reimplementation. Falls back
    to the normal read for checkpoints that predate torch's zipfile format (they
    cannot be mapped).
    """

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Any] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if not isinstance(fs, LocalFileSystem):
            # mmap needs a real local file; let pl_load handle fsspec/URL paths.
            return pl_load(path, map_location=map_location, weights_only=weights_only)
        try:
            return torch.load(
                path, map_location=map_location, weights_only=weights_only, mmap=True
            )
        except (RuntimeError, ValueError, NotImplementedError) as e:
            log.warning(
                f"mmap load of {path} failed ({e}); falling back to a full read"
            )
            return pl_load(path, map_location=map_location, weights_only=weights_only)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()
    action_timing = _action_timing_from_config(cfg)
    data_contracts = local_data_contracts(cfg)
    validate_model_timing(action_timing, cfg.model)
    local_checkpoint = None
    if data_contracts:
        checkpoint_path = _effective_local_checkpoint_path(cfg)
        if checkpoint_path:
            local_checkpoint = MmapCheckpointIO().load_checkpoint(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            ModelWrapper.validate_checkpoint_contracts(
                local_checkpoint, _build_model_config_tree(cfg)
            )

    # One SQL pull / path resolution per dataset spec across train, valid and
    # the norm-stat copies; dropped on exit so nothing outlives this run.
    with resolve_once():
        train_datasets = {}
        for dataset_name in cfg.data.train_datasets:
            train_datasets[dataset_name] = _instantiate_dataset(
                cfg.data.train_datasets[dataset_name], dataset_name=dataset_name
            )

        valid_datasets = {}
        for dataset_name in cfg.data.valid_datasets:
            valid_datasets[dataset_name] = _instantiate_dataset(
                cfg.data.valid_datasets[dataset_name], dataset_name=dataset_name
            )

        validate_dataset_timing(action_timing, train_datasets, valid_datasets)
        validate_recording_split(train_datasets, valid_datasets)
        for name, timing in action_timing.items():
            log.info(
                f"Action timing {name}: {timing.source_fps:g} Hz, "
                f"H={timing.source_horizon}, stride={timing.action_stride}, "
                f"output={timing.output_chunk_length}, includes_current=True; "
                f"source span={timing.source_span_s:.9f}s, "
                f"effective span={timing.effective_span_s:.9f}s"
            )

        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        assert (
            "MultiDataModuleWrapper" in cfg.data._target_
        ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
        datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
        )

        # Stats-only MultiDataset (no graph of its own; explicitly populated from
        # datamodule.train_datasets). MultiDataset now owns NormStats's role too.
        norm_stats = MultiDataset(
            state={},
            norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
        )
        norm_stats.populate_from_datasets(datamodule.train_datasets)

        from egomimic.rldb.zarr import norm_cache

        sample_frac = OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0)
        explicit_path = OmegaConf.select(
            cfg, "norm_stats.precomputed_norm_path", default=None
        )
        # Code default is False so configs without the key keep the old behaviour
        # (always recompute); the shipped configs set norm_stats.use_cache=true.
        use_cache = bool(OmegaConf.select(cfg, "norm_stats.use_cache", default=False))
        cache_dir = OmegaConf.select(cfg, "norm_stats.cache_dir", default=None)

        for dataset_name, dataset in datamodule.train_datasets.items():
            log.info(f"Inferring shapes for dataset <{dataset_name}>")
            norm_stats.infer_shapes_from_batch(dataset[0])
            if local_checkpoint is not None:
                # Resume/eval must retain the numerical coordinate system that
                # the saved weights learned, regardless of current data/cache.
                continue
            instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
            keymap_cfg = instantiate_copy.resolver.key_map
            km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

            # this remove annotation and image keys from the keymap
            km["norm_mode"] = True

            instantiate_copy.resolver.key_map = km
            norm_dataset = _instantiate_dataset(
                instantiate_copy, dataset_name=dataset_name
            )

            emb = get_embodiment_id(dataset_name)
            key = inputs = cached = None
            if explicit_path is None and use_cache and cache_dir:
                episodes = {
                    h: ds.normalization_fingerprint()
                    if hasattr(ds, "normalization_fingerprint")
                    else norm_cache.episode_fingerprint(
                        getattr(ds, "episode_path", None)
                    )
                    for h, ds in dataset.datasets.items()
                }
                inputs = norm_cache.cache_inputs(
                    dataset_name,
                    episodes,
                    cfg.data.train_datasets[dataset_name],
                    sample_frac,
                )
                key = norm_cache.norm_cache_key(inputs)
                cached = norm_cache.find_cached(cache_dir, dataset_name, key, emb)
                if cached is not None:
                    log.info(f"norm stats for <{dataset_name}>: cache hit {cached}")

            # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
            norm_stats.infer_norm_from_dataset(
                norm_dataset,
                dataset_name,
                sample_frac=sample_frac,
                num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
                precomputed_norm_path=explicit_path
                if explicit_path is not None
                else cached,
            )
            if key is not None and cached is None:
                if norm_stats.norm_stats.get(emb):
                    norm_cache.write_cached(
                        cache_dir,
                        dataset_name,
                        key,
                        inputs,
                        emb,
                        norm_stats.norm_stats[emb],
                        norm_stats._norm_run_metadata,
                    )
            # Cache norm stats if save_cache_dir is set
            save_cache_dir = OmegaConf.select(
                cfg, "norm_stats.save_cache_dir", default=None
            )
            if save_cache_dir:
                norm_stats.cache_stats(save_cache_dir=save_cache_dir)

        if local_checkpoint is not None:
            norm_stats = _restore_local_norm_state(local_checkpoint, norm_stats)
            log.info(
                "Restored checkpoint normalization for all train/validation datasets; skipped recomputation"
            )

    # Wire each training/valid MultiDataset to the stats-only ``norm_stats``
    # by reference. Bounds-check + normalize run at the MultiDataset level in
    # ``__getitem__`` — not as per-leaf transforms — which avoids the shared
    # transform_list aliasing trap.
    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    for ds in datamodule.valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    _log_dataset_frame_counts(datamodule.train_datasets, datamodule.valid_datasets)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Resolve mode: support both new `mode` key and legacy `train`/`eval` booleans
    if cfg.get("mode") is not None:
        mode = cfg.mode
    elif cfg.get("train", False):
        mode = "train"
    elif cfg.get("eval", False):
        mode = "eval"
    else:
        raise ValueError("Config must specify either `mode` or `train`/`eval` booleans")

    # In eval mode, apply trainer overrides from the eval object and disable logger
    if mode == "eval":
        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        log.info(
            "Eval mode: applying trainer overrides from eval config, disabling logger"
        )
        with open_dict(cfg):
            for k, v in eval_obj.override_dict.items():
                cfg.trainer[k] = v
            cfg.trainer.devices = 1
            cfg.trainer.num_nodes = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.logger = None

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if cfg.get("mmap_checkpoint", True):
        plugins.append(MmapCheckpointIO())
    if os.environ.get("SLURM_JOB_ID"):
        # requeue_signal is a single signal, not a list -- SignalConnector passes
        # it straight to signal.getsignal(), which raises TypeError on a list.
        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR1))
        print("SLURM REQUEUE ENABLED")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins or None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (
        not data_contracts
        and os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
        cfg.ckpt_path = last_ckpt_path

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj

        if local_checkpoint is not None:
            # The legacy path below loads weights only. Local recipes also
            # verify the time/data/norm contract and require an exact network.
            model.on_load_checkpoint(local_checkpoint)
            model.load_state_dict(local_checkpoint["state_dict"], strict=True)
            log.info(
                f"Loaded local checkpoint weights and contracts from {cfg.ckpt_path}"
            )

        if hasattr(eval_obj, "run"):
            eval_obj.run(trainer, model, datamodule, cfg)
        else:
            # Default: load checkpoint + validate (unchanged from main)
            ckpt_path = cfg.get("ckpt_path")
            if ckpt_path and local_checkpoint is None:
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                log.info(f"Loaded weights from {ckpt_path}")
            log.info("Starting evaluation!")
            trainer.validate(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    test_metrics = {}  # my stub
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    # cfg = OmegaConf.resolve(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()

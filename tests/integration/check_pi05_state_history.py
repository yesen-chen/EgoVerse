"""Real OpenPI CPU history transforms, token budget, RTC prefix and LoRA shapes.

Runs in an existing Pi0.5 environment; allocates no model weights and touches no
robot. Optional --tokenizer-path exercises the local production tokenizer.
"""

import argparse
import copy
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pi05/scripts"))
from _bootstrap import setup  # noqa: E402


def run_checks(openpi_root=None, tokenizer_path=None):
    setup(cpu=True, openpi_root=openpi_root)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from egoverse_pi05 import config as local
    from egoverse_pi05.assets import CAMERAS
    from egoverse_pi05.rtc import SCHEMA, RtcInference
    from openpi import transforms
    from openpi.models import tokenizer
    from openpi.shared import normalize
    from openpi.training import data_loader

    from egomimic.robot.piper_pico.state_history import make_history_contract

    def wrapped_stats(upper):
        return {
            "norm_stats": {
                key: {
                    "mean": [0.0] * 14,
                    "std": [1.0] * 14,
                    "q01": [0.0] * 14,
                    "q99": [float(upper)] * 14,
                }
                for key in ("state", "actions")
            }
        }

    domains = {"piper": wrapped_stats(4), "ego": wrapped_stats(8)}
    stats = {
        key: normalize.NormStats(**{k: np.asarray(v) for k, v in fields.items()})
        for key, fields in domains["piper"]["norm_stats"].items()
    }

    def sample(count, domain="piper", mode="width"):
        data = {
            "observation.state": np.linspace(0.1, 1.4, 14, dtype=np.float32),
            "actions": np.full((100, 14), 2.0, np.float32),
            "prompt": "Pick up object",
            "domain": domain,
            "gripper_mode": mode,
        }
        if count:
            data["observation.state_history"] = (
                np.arange(count * 14, dtype=np.float32).reshape(count, 14) / 20
            )
        for name in CAMERAS if domain == "piper" else ["front_1"]:
            data[f"observation.images.{name}"] = np.zeros((12, 16, 3), np.uint8)
        return data

    class SpySentencePiece:
        def __init__(self):
            self.texts = []

        def encode(self, text, add_bos=False):
            self.texts.append(text)
            return ([1] if add_bos else []) + [
                sum(map(ord, word)) % 20000 + 3 for word in text.split()
            ]

    def tokenizer_at(length):
        if tokenizer_path is not None:
            return local.LocalTokenizer(tokenizer_path, length)
        tok = object.__new__(local.LocalTokenizer)
        tok._max_len, tok._tokenizer = length, SpySentencePiece()
        return tok

    class HistoryChecks(unittest.TestCase):
        def test_real_data_factory_single_and_mixed_batches_consume_history(self):
            history = make_history_contract([-0.07, -0.025])
            for mixed in (False, True):
                module = types.ModuleType("pi05_training_data")
                module.load_training_manifest = lambda path: {
                    "schema": "egoverse_pi05_mixed_manifest_v1"
                    if mixed
                    else "egoverse_pi05_manifest_v1",
                    "state_history": history,
                }
                module.load_training_stats = (
                    lambda manifest, path: domains
                    if mixed
                    else {"piper": domains["piper"]}
                )
                with tempfile.TemporaryDirectory() as directory:
                    Path(directory, "norm_stats.json").write_text(
                        json.dumps(domains["piper"])
                    )
                    with (
                        patch.dict(
                            sys.modules,
                            {"egomimic.robot.piper_pico.pi05_training_data": module},
                        ),
                        patch.object(
                            local, "LocalTokenizer", return_value=tokenizer_at(512)
                        ),
                        patch.object(local, "validate_manifest_contract"),
                    ):
                        cfg = local.EgoVerseDataFactory(stats_dir=directory).create(
                            None,
                            local.make_model_config(
                                "lora", state_history=history, max_token_len=512
                            ),
                        )
                    self.assertEqual(
                        set(cfg.norm_stats),
                        set() if mixed else {"state", "actions", "state_history"},
                    )
                    samples = (
                        [sample(2), sample(2, "ego", "zero")]
                        if mixed
                        else [sample(2), sample(2)]
                    )
                    result = data_loader.transform_dataset(samples, cfg)
                    batch = data_loader._collate_fn([result[0], result[1]])
                    self.assertNotIn("state_history", batch)
                    self.assertEqual(batch["actions"].shape, (2, 100, 32))
                    self.assertEqual(batch["state"].shape, (2, 32))
                    self.assertEqual(batch["tokenized_prompt"].shape, (2, 512))
                    if mixed:
                        self.assertFalse(batch["state"][1, [6, 13]].any())
                        self.assertFalse(batch["actions"][1, :, [6, 13]].any())
                    restored = pickle.loads(pickle.dumps(cfg.data_transforms.inputs))
                    self.assertEqual(restored[0].state_history, history)

        def test_input_counts_shapes_and_old_mode_reject_extra(self):
            for count in (0, 1, 3):
                contract = make_history_contract(
                    [-0.025 * i for i in range(count, 0, -1)]
                )
                result = local.BimanualInputs(state_history=contract)(sample(count))
                self.assertEqual(result["state"].shape, (14,))
                self.assertEqual(result["actions"].shape, (100, 14))
                if count:
                    self.assertEqual(result["state_history"].shape, (count, 14))
                    bad = sample(count)
                    bad.pop("observation.state_history")
                    with self.assertRaises(ValueError):
                        local.BimanualInputs(state_history=contract)(bad)
            with self.assertRaisesRegex(ValueError, "not enabled"):
                local.BimanualInputs()(sample(1))

        def test_same_scale_history_and_current_for_single_and_mixed(self):
            history = make_history_contract([-0.05, -0.025])
            for domain, mode, upper in (
                ("piper", "width", 4),
                ("ego", "width", 8),
                ("ego", "zero", 8),
            ):
                raw = sample(2, domain, mode)
                physical = local.BimanualInputs(allow_ego=True, state_history=history)(
                    raw
                )
                mixed = local.DomainNormalize(domains)(physical)
                expected = raw["observation.state_history"] / (upper + 1e-6) * 2 - 1
                if mode == "zero":
                    expected[:, [6, 13]] = 0
                np.testing.assert_allclose(mixed["state_history"], expected, atol=1e-7)
                if domain == "piper":
                    standalone = transforms.Normalize(
                        local.with_history_stats(stats), use_quantiles=True
                    )(physical)
                    for key in ("state", "state_history", "actions"):
                        np.testing.assert_allclose(
                            standalone[key], mixed[key], atol=1e-7
                        )

        def test_token_order_consumption_padding_and_native_nohistory_parity(self):
            tok = tokenizer_at(512)
            physical = local.BimanualInputs(
                state_history=make_history_contract([-0.05, -0.025])
            )(sample(2))
            normalized = transforms.Normalize(
                local.with_history_stats(stats), use_quantiles=True
            )(physical)
            result = local.TokenizeStateHistory(tok, 2)(normalized)
            self.assertNotIn("state_history", result)
            self.assertNotIn("prompt", result)
            np.testing.assert_array_equal(result["state"], normalized["state"])
            np.testing.assert_array_equal(result["actions"], normalized["actions"])
            padded = transforms.PadStatesAndActions(32)(result)
            self.assertEqual(padded["actions"].shape, (100, 32))
            self.assertFalse(padded["actions"][:, 14:].any())
            self.assertTrue(result["tokenized_prompt_mask"].any())
            baseline = object.__new__(tokenizer.PaligemmaTokenizer)
            baseline._max_len, baseline._tokenizer = tok._max_len, tok._tokenizer
            for left, right in zip(
                tok.tokenize("Pick_up\nobject", normalized["state"]),
                baseline.tokenize("Pick_up\nobject", normalized["state"]),
                strict=True,
            ):
                np.testing.assert_array_equal(left, right)
            spy = tokenizer_at(512)
            spy._tokenizer = SpySentencePiece()
            local.TokenizeStateHistory(spy, 2)(normalized)

            def bins(row):
                return " ".join(
                    map(str, np.digitize(row, np.linspace(-1, 1, 257)[:-1]) - 1)
                )

            self.assertIn(
                "History (oldest first): "
                + "; ".join(bins(row) for row in normalized["state_history"]),
                spy._tokenizer.texts[-1],
            )
            with self.assertRaisesRegex(ValueError, "Refusing to truncate"):
                tokenizer_at(8).tokenize(
                    "Pick",
                    normalized["state"],
                    state_history=normalized["state_history"],
                )

        def test_legacy_long_prompt_keeps_upstream_truncation_and_dtypes(self):
            tok = tokenizer_at(8)
            baseline = object.__new__(tokenizer.PaligemmaTokenizer)
            baseline._max_len, baseline._tokenizer = tok._max_len, tok._tokenizer
            prompt = "Pick up the object and place it in the container. " * 20
            for state in (None, np.linspace(-1, 1, 14)):
                for actual, expected in zip(
                    tok.tokenize(prompt, state),
                    baseline.tokenize(prompt, state),
                    strict=True,
                ):
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, expected.dtype)

        def test_mixed_history_tokens_match_serving_at_bin_boundaries(self):
            contract = make_history_contract([-0.025])
            # Serving produces 0.4999999999999998. A premature float32 cast
            # would turn it into 0.5 and move current/history to the next bin.
            boundary_stats = wrapped_stats(1.3333333333333335 - 1e-6)
            boundary_normalizer = {
                key: normalize.NormStats(
                    **{field: np.asarray(value) for field, value in fields.items()}
                )
                for key, fields in boundary_stats["norm_stats"].items()
            }
            raw = sample(1)
            raw["observation.state"][:] = 1
            raw["observation.state_history"][:] = 1
            physical = local.BimanualInputs(allow_ego=True, state_history=contract)(raw)
            mixed_normalizer = local.DomainNormalize(
                {domain: boundary_stats for domain in ("piper", "ego")}
            )
            mixed = mixed_normalizer(physical)
            serving = transforms.Normalize(
                local.with_history_stats(boundary_normalizer), use_quantiles=True
            )(physical)
            for key in ("state", "state_history"):
                np.testing.assert_array_equal(mixed[key], serving[key])
                self.assertEqual(mixed[key].dtype, serving[key].dtype)
            token_transform = local.TokenizeStateHistory(tokenizer_at(256), 1)
            np.testing.assert_array_equal(
                token_transform(mixed)["tokenized_prompt"],
                token_transform(serving)["tokenized_prompt"],
            )
            self.assertEqual(mixed["actions"].dtype, np.float32)
            legacy = local.BimanualInputs(allow_ego=True)(sample(0))
            self.assertEqual(mixed_normalizer(legacy)["state"].dtype, np.float32)

        def test_training_serving_rtc_share_transform_and_unchanged_action_prefix(self):
            contract = make_history_contract([-0.07, -0.025])
            tok = tokenizer_at(512)
            pipeline = transforms.compose(
                [
                    local.BimanualInputs(state_history=contract),
                    transforms.Normalize(
                        local.with_history_stats(stats), use_quantiles=True
                    ),
                    local.TokenizeStateHistory(tok, 2),
                    transforms.PadStatesAndActions(32),
                ]
            )
            raw = sample(2)
            expected = pipeline(copy.deepcopy(raw))
            seen = {}

            class Policy:
                _rng = jax.random.key(0)
                _sample_kwargs = {"num_steps": 10}
                _input_transform = staticmethod(pipeline)
                _output_transform = staticmethod(lambda data: data)

                def infer(self, observation, **kw):
                    seen["baseline"] = pipeline(copy.deepcopy(observation))
                    return {"actions": np.zeros((100, 14))}

            def sampler(key, observation, previous, weights, **kw):
                seen.update(
                    observation=observation,
                    previous=np.asarray(previous),
                    weights=np.asarray(weights),
                )
                return jnp.asarray(previous)

            obs = {k: v for k, v in raw.items() if k != "actions"}
            runner = RtcInference(Policy(), sampler)
            request = {
                "schema": SCHEMA,
                "request_id": 1,
                "previous_actions": raw["actions"],
                "weights": np.ones(100, np.float32),
            }
            runner.infer(obs, request)
            np.testing.assert_array_equal(
                seen["previous"][0], expected["actions"].astype(np.float32)
            )
            np.testing.assert_array_equal(
                np.asarray(seen["observation"].tokenized_prompt)[0],
                expected["tokenized_prompt"],
            )
            self.assertFalse(seen["weights"][..., 14:].any())
            runner.infer(obs, {**request, "weights": np.zeros(100, np.float32)})
            np.testing.assert_array_equal(
                seen["baseline"]["tokenized_prompt"], expected["tokenized_prompt"]
            )

        def test_model_shape_and_history_token_budget_config(self):
            contract = make_history_contract([-0.07, -0.025])
            for mode in ("lora", "full"):
                model = local.make_model_config(mode, state_history=contract)
                self.assertEqual(
                    (model.action_dim, model.action_horizon, model.max_token_len),
                    (32, 100, 256),
                )
                self.assertEqual(local.make_model_config(mode).max_token_len, 200)
                self.assertEqual(
                    local.make_model_config(
                        mode, state_history=contract, max_token_len=512
                    ).max_token_len,
                    512,
                )
            # LoRA shape-only path is exercised by check_pi05_lora.py --inspect;
            # no hidden state dimension or action-head expansion is introduced.

        def test_equal_current_and_history_do_not_cross_different_token_bins(self):
            tok = tokenizer_at(256)
            tok._tokenizer = SpySentencePiece()
            # The value is just below a bin edge in the normalizer's float64.
            # Rounding only history to float32 would incorrectly change its bin.
            state = np.full(14, np.nextafter(0.5, -np.inf))
            local.TokenizeStateHistory(tok, 1)(
                {"state": state, "state_history": state[None], "prompt": "Pick"}
            )
            text = tok._tokenizer.texts[-1]
            expected = " ".join(
                map(str, np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1)
            )
            self.assertIn(
                f"State: {expected}; History (oldest first): {expected};", text
            )

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(HistoryChecks)
    )
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root")
    parser.add_argument("--tokenizer-path")
    args = parser.parse_args()
    sys.exit(0 if run_checks(args.openpi_root, args.tokenizer_path) else 1)

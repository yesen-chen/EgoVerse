"""CPU-only OpenPI input contract checks; no model weights or network access.

Run in the user-managed Pi0.5 interpreter. The regular unit environment does
not need JAX; test_pi05_mixed_inputs.py supplies the optional subprocess entry.
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


def run_checks(openpi_root=None):
    setup(cpu=True, openpi_root=openpi_root)
    import numpy as np
    from egoverse_pi05 import config as local
    from egoverse_pi05.assets import CAMERAS
    from openpi import transforms
    from openpi.models import tokenizer
    from openpi.training import data_loader

    def stats(upper):
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

    domains = {"piper": stats(4), "ego": stats(8)}

    def sample(domain="ego", mode="width"):
        result = {
            "observation.state": np.full(14, 2.0, dtype=np.float32),
            "actions": np.full((100, 14), 2.0, dtype=np.float32),
            "prompt": "pick up the object",
            "domain": domain,
            "gripper_mode": mode,
        }
        result["observation.state"][[6, 13]] = 0.8
        result["actions"][:, [6, 13]] = 0.8
        for camera in CAMERAS if domain == "piper" else ("front_1",):
            result[f"observation.images.{camera}"] = np.full(
                (12, 16, 3), 123, dtype=np.uint8
            )
        return result

    class RecordingSentencePiece:
        def __init__(self):
            self.texts = []

        def encode(self, text, add_bos=False):
            self.texts.append(text)
            return [1, 2, 3]

    class Samples:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return copy.deepcopy(self.samples[index])

    class MixedInputTests(unittest.TestCase):
        def setUp(self):
            self.input = local.BimanualInputs(allow_ego=True)
            self.norm = local.DomainNormalize(domains)

        def test_default_serving_input_rejects_ego_and_missing_wrist(self):
            with self.assertRaisesRegex(ValueError, "mixed training"):
                local.BimanualInputs()(sample())
            piper = sample("piper")
            del piper["observation.images.left_wrist"]
            with self.assertRaises(KeyError):
                local.BimanualInputs()(piper)

        def test_ego_masked_slots_keep_front_image(self):
            result = self.input(sample())
            self.assertTrue(result["image_mask"]["base_0_rgb"])
            for slot in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
                self.assertFalse(result["image_mask"][slot])
                self.assertFalse(result["image"][slot].any())
            self.assertTrue((result["image"]["base_0_rgb"] == 123).all())

        def test_piper_masks_remain_all_valid(self):
            result = self.input(sample("piper"))
            self.assertTrue(all(result["image_mask"].values()))

        def test_domains_use_different_quantiles_for_same_physical_value(self):
            for domain, upper in (("piper", 4.0), ("ego", 8.0)):
                original = sample(domain)
                result = self.norm(self.input(original))
                np.testing.assert_allclose(
                    result["state"],
                    original["observation.state"] / (upper + 1e-6) * 2 - 1,
                    atol=1e-7,
                )
                self.assertNotIn("domain", result)
                self.assertNotIn("gripper_mode", result)
                np.testing.assert_array_equal(original["actions"][:, 0], 2.0)

        def test_zero_gripper_is_inserted_after_normalization(self):
            result = self.norm(self.input(sample(mode="zero")))
            np.testing.assert_array_equal(result["state"][[6, 13]], 0.0)
            np.testing.assert_array_equal(result["actions"][:, [6, 13]], 0.0)
            self.assertLess(result["state"][0], -0.49)

        def test_width_is_not_rescaled_or_zeroed_again(self):
            result = self.norm(self.input(sample()))
            np.testing.assert_allclose(result["state"][[6, 13]], -0.8, atol=1e-6)
            np.testing.assert_allclose(result["actions"][:, [6, 13]], -0.8, atol=1e-6)

        def test_domain_transform_survives_worker_serialization(self):
            restored = pickle.loads(pickle.dumps(self.norm))
            expected = self.norm(self.input(sample(mode="zero")))
            result = restored(self.input(sample(mode="zero")))
            np.testing.assert_array_equal(result["state"], expected["state"])
            np.testing.assert_array_equal(result["actions"], expected["actions"])

        def test_invalid_domain_modes_and_unexpected_ego_wrists_rejected(self):
            for domain, mode in (("other", "width"), ("piper", "zero"), ("ego", "bad")):
                data = sample()
                data.update(domain=domain, gripper_mode=mode)
                with self.assertRaises(ValueError):
                    self.input(data)
            data = sample()
            data["observation.images.left_wrist"] = data["observation.images.front_1"]
            with self.assertRaisesRegex(ValueError, "only provide"):
                self.input(data)
            with self.assertRaisesRegex(ValueError, "separate"):
                local.DomainNormalize({"piper": stats(4)})

        def test_actual_openpi_pipeline_normalizes_once_before_state_tokenizer(self):
            # Use the actual Paligemma state discretization and TokenizePrompt,
            # while replacing only the local sentence-piece file with a spy.
            tok = object.__new__(tokenizer.PaligemmaTokenizer)
            tok._max_len = 200
            tok._tokenizer = RecordingSentencePiece()
            training_module = types.ModuleType("pi05_training_data")
            training_module.load_training_manifest = lambda path: {
                "schema": "egoverse_pi05_mixed_manifest_v1"
            }
            training_module.load_training_stats = lambda manifest, path: domains
            with (
                patch.dict(
                    sys.modules,
                    {"egomimic.robot.piper_pico.pi05_training_data": training_module},
                ),
                patch.object(local, "LocalTokenizer", return_value=tok),
            ):
                cfg = local.EgoVerseDataFactory(
                    manifest_path="unused.json",
                    stats_dir="unused",
                    tokenizer_path="unused",
                ).create(None, local.make_model_config("lora"))
            self.assertEqual(cfg.norm_stats, {})
            dataset = data_loader.transform_dataset(
                Samples([sample("piper"), sample(mode="zero"), sample()]), cfg
            )
            for index, (domain, mode) in enumerate(
                (("piper", "width"), ("ego", "zero"), ("ego", "width"))
            ):
                result = dataset[index]
                expected = self.norm(self.input(sample(domain, mode)))["state"]
                np.testing.assert_allclose(result["state"][:14], expected)
                self.assertEqual(result["state"].shape, (32,))
                self.assertEqual(result["actions"].shape, (100, 32))
                np.testing.assert_array_equal(result["state"][14:], 0.0)
                np.testing.assert_array_equal(result["actions"][:, 14:], 0.0)
                self.assertTrue(result["tokenized_prompt_mask"].any())
                bins = np.digitize(expected, np.linspace(-1, 1, 257)[:-1]) - 1
                state_text = " ".join(map(str, bins))
                self.assertIn(f"State: {state_text};", tok._tokenizer.texts[-1])
                for image in result["image"].values():
                    self.assertEqual(image.shape, (224, 224, 3))
                if domain == "ego":
                    self.assertFalse(result["image_mask"]["left_wrist_0_rgb"])

        def test_single_piper_factory_keeps_generic_normalization(self):
            training_module = types.ModuleType("pi05_training_data")
            training_module.load_training_manifest = lambda path: {
                "schema": "egoverse_pi05_manifest_v1"
            }
            training_module.load_training_stats = lambda manifest, path: {
                "piper": domains["piper"]
            }
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "norm_stats.json").write_text(
                    json.dumps(domains["piper"])
                )
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "egomimic.robot.piper_pico.pi05_training_data": training_module
                        },
                    ),
                    patch.object(local, "validate_manifest_contract"),
                    patch.object(local, "model_transforms", return_value=[]),
                ):
                    cfg = local.EgoVerseDataFactory(stats_dir=directory).create(
                        None, local.make_model_config("lora")
                    )
                self.assertEqual(len(cfg.data_transforms.inputs), 1)
                self.assertEqual(set(cfg.norm_stats), {"state", "actions"})
                result = data_loader.transform_dataset(Samples([sample("piper")]), cfg)[
                    0
                ]
                expected = self.norm(self.input(sample("piper")))["state"]
                np.testing.assert_allclose(result["state"], expected, atol=1e-7)

        def test_ego_and_piper_make_one_actual_openpi_batch(self):
            # Stack the outputs with upstream collate; no dataset or GPU needed.
            config = local.EgoVerseDataConfig(
                repo_id="local/mixed-test",
                norm_stats={},
                use_quantile_norm=True,
                data_transforms=transforms.Group(inputs=[self.input, self.norm]),
                model_transforms=transforms.Group(
                    inputs=[transforms.PadStatesAndActions(32)]
                ),
            )
            dataset = data_loader.transform_dataset(
                Samples([sample("piper"), sample()]), config
            )
            batch = data_loader._collate_fn([dataset[0], dataset[1]])
            self.assertEqual(batch["actions"].shape, (2, 100, 32))
            np.testing.assert_array_equal(
                batch["image_mask"]["left_wrist_0_rgb"], [True, False]
            )
            self.assertTrue(np.isfinite(batch["actions"]).all())

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MixedInputTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root")
    args = parser.parse_args()
    sys.exit(0 if run_checks(args.openpi_root) else 1)

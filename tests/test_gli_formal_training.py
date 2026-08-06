from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import Dataset


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "LeFusion"))

from checkpointing import (  # noqa: E402
    GLI_TRAINING_CHECKPOINT_SCHEMA,
    canonical_config_hash,
    load_training_checkpoint,
)
from dataset.gli_sampler import GLIStratifiedSampler  # noqa: E402
from ddpm import GaussianDiffusion_Nolatent, Trainer  # noqa: E402
from train.tracking import validate_wandb_config  # noqa: E402
from train.train import validate_training_config  # noqa: E402
from train.validation import EarlyStopping, run_gli_validation  # noqa: E402
from scripts.gli_formal_training_preflight import build_preflight_cfg  # noqa: E402


def _records():
    records = []
    for label in (1, 2, 3, 4):
        for role in ("interior", "boundary"):
            for subject in ("subject_a", "subject_b"):
                for patch in range(2):
                    records.append(
                        {
                            "anchor_label": str(label),
                            "sample_role": role,
                            "subject_id": f"{subject}_{label}_{role}",
                            "relative_path": f"{label}_{role}_{subject}_{patch}.npz",
                        }
                    )
    return records


class _ZeroDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, time, cond=None, **kwargs):
        return torch.zeros_like(x) + self.scale


class _TinyDataset(Dataset):
    def __init__(self):
        self.records = _records()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        label = int(record["anchor_label"])
        mask = torch.zeros((4, 1, 1, 1))
        mask[label - 1] = 1
        return {
            "data": torch.zeros((4, 1, 1, 1)),
            "lesion_mask": mask,
            "label": torch.tensor([[[[label]]]], dtype=torch.int64),
            "hist": torch.zeros(64),
            "subject_id": record["subject_id"],
            "anchor_label": label,
            "sample_index": index,
        }


def _tiny_diffusion():
    return GaussianDiffusion_Nolatent(
        _ZeroDenoiser(),
        image_size=1,
        num_frames=1,
        spatial_shape=(1, 1, 1),
        channels=4,
        timesteps=4,
        loss_type="l1",
        device=torch.device("cpu"),
        data_type="gli",
    )


def _metadata():
    return {
        "experiment_id": "exp005",
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
        "manifest_hash": "c" * 64,
        "split_hash": "d" * 64,
        "data_type": "gli",
        "patch_size_xyz": [1, 1, 1],
        "spatial_shape_dhw": [1, 1, 1],
        "sampler_name": "gli_anchor_role_subject",
        "wandb_run_id": "exp005-test",
    }


class GLIFormalTrainingTests(unittest.TestCase):
    def test_sampler_balances_strata_and_subjects_deterministically(self):
        records = _records()
        sampler = GLIStratifiedSampler(records, seed=17, num_samples=80)
        indices = sampler.indices_for_epoch(0)
        self.assertEqual(indices, sampler.indices_for_epoch(0))
        self.assertNotEqual(indices, sampler.indices_for_epoch(1))
        strata = Counter(
            (int(records[index]["anchor_label"]), records[index]["sample_role"])
            for index in indices
        )
        self.assertEqual(set(strata.values()), {10})
        for stratum in strata:
            subjects = Counter(
                records[index]["subject_id"]
                for index in indices
                if (int(records[index]["anchor_label"]), records[index]["sample_role"])
                == stratum
            )
            self.assertLessEqual(max(subjects.values()) - min(subjects.values()), 1)

    def test_validation_records_total_channels_units_and_coverage(self):
        metrics = run_gli_validation(
            _tiny_diffusion(),
            _TinyDataset(),
            device=torch.device("cpu"),
            batch_size=4,
            num_workers=0,
            seed=123,
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["val/ema/total_loss"])))
        self.assertEqual(metrics["val/ema/effective_units"], 32)
        self.assertEqual(metrics["val/ema/anchor_label_coverage"], 4)
        for name in ("netc", "snfh", "et", "rc"):
            self.assertEqual(metrics[f"val/ema/{name}_effective_units"], 8)

    def test_early_stopping_uses_relative_improvement_and_patience(self):
        stopper = EarlyStopping(patience=2, relative_min_delta=0.01, warmup_steps=10)
        self.assertEqual(stopper.update(1.0, 5), (True, False))
        self.assertEqual(stopper.update(0.995, 10), (False, False))
        self.assertEqual(stopper.update(0.994, 12), (False, True))
        self.assertEqual(stopper.update(0.98, 14), (True, False))

    def test_full_checkpoint_round_trip_restores_training_state(self):
        dataset = _TinyDataset()
        sampler = GLIStratifiedSampler(dataset.records, seed=7)
        cfg = OmegaConf.create(
            {"seed": 7, "sampler": {"drop_last": False}, "model": {"results_folder": "x"}}
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                _tiny_diffusion(),
                cfg=cfg,
                dataset=dataset,
                train_batch_size=4,
                gradient_accumulate_every=1,
                train_num_steps=2,
                num_workers=0,
                device=torch.device("cpu"),
                train_sampler=sampler,
                results_folder=directory,
                checkpoint_metadata=_metadata(),
                resolved_config={"seed": 7},
            )
            trainer.step = 1
            trainer.train_epoch = 2
            trainer.batch_in_epoch = 3
            trainer._reset_train_iterator(skip_batches=trainer.batch_in_epoch)
            path = Path(directory) / "latest.pt"
            trainer.save_checkpoint(path, kind="latest")
            expected_next_indices = trainer._next_train_batch()["sample_index"].tolist()
            payload = load_training_checkpoint(path, expected_metadata=_metadata())
            self.assertEqual(payload["schema_version"], GLI_TRAINING_CHECKPOINT_SCHEMA)

            restored = Trainer(
                _tiny_diffusion(),
                cfg=cfg,
                dataset=dataset,
                train_batch_size=4,
                gradient_accumulate_every=1,
                train_num_steps=2,
                num_workers=0,
                device=torch.device("cpu"),
                train_sampler=GLIStratifiedSampler(dataset.records, seed=7),
                results_folder=directory,
                checkpoint_metadata=_metadata(),
                resolved_config={"seed": 7},
            )
            restored.load(path)
            self.assertEqual(restored.step, 1)
            self.assertEqual(restored.train_epoch, 2)
            self.assertEqual(restored.batch_in_epoch, 3)
            self.assertEqual(
                restored._next_train_batch()["sample_index"].tolist(), expected_next_indices
            )

    def test_formal_configs_are_complete_and_batch_is_preflight_adjustable(self):
        config_dir = str(ROOT / "LeFusion" / "train" / "config")
        for name, expected_batch in (
            ("gli_formal_64x64x32", (4, 1)),
            ("gli_formal_80x96x80", (1, 4)),
            ("gli_exp008_conditional_inpainting_64x64x32", (4, 1)),
        ):
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name="base_cfg", overrides=[f"+experiment={name}"])
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            validate_training_config(cfg)
            validate_wandb_config(cfg)
            self.assertIn("preflight", cfg)
            self.assertNotEqual(cfg.preflight.run_id, cfg.wandb.run_id)
            self.assertIn("preflight", cfg.preflight.run_id)
            self.assertEqual(
                (cfg.model.batch_size, cfg.model.gradient_accumulate_every), expected_batch
            )
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            fallback = compose(
                config_name="base_cfg",
                overrides=[
                    "+experiment=gli_formal_64x64x32",
                    "model.batch_size=2",
                    "model.gradient_accumulate_every=2",
                ],
            )
        validate_training_config(fallback)

    def test_config_hash_ignores_resume_transport_only(self):
        first = {
            "checkpoint": {"resume_from": None, "latest_every_steps": 500},
            "wandb": {"resume": "never", "run_id": "same"},
        }
        second = {
            "checkpoint": {"resume_from": "latest.pt", "latest_every_steps": 500},
            "wandb": {"resume": "must", "run_id": "same"},
        }
        self.assertEqual(canonical_config_hash(first), canonical_config_hash(second))

    def test_preflight_identity_is_generic_and_separate_from_formal_run(self):
        cfg = OmegaConf.create(
            {
                "wandb": {"run_id": "exp008-p64", "run_name": "exp008-p64", "dir": "x", "resume": "never"},
                "model": {"results_folder": "formal"},
                "preflight": {
                    "run_id": "exp008-p64-preflight",
                    "run_name": "exp008-p64-preflight",
                    "output_dir": "preflight-output",
                },
            }
        )
        cloned = build_preflight_cfg(cfg)
        self.assertEqual(cloned.wandb.run_id, "exp008-p64-preflight")
        self.assertNotEqual(cloned.wandb.run_id, cfg.wandb.run_id)
        cfg.preflight.run_id = "exp008-p64"
        with self.assertRaisesRegex(ValueError, "preflight ID"):
            build_preflight_cfg(cfg)


if __name__ == "__main__":
    unittest.main()

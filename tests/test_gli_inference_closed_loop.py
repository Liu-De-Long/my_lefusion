from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "LeFusion"))

from checkpointing import load_diffusion_checkpoint  # noqa: E402
from dataset.gli_hist_in import (  # noqa: E402
    GLIInferenceDataset,
    build_explicit_brain_support,
    extract_xyz_patch,
)
from ddpm import (  # noqa: E402
    GaussianDiffusion_Nolatent,
    compose_gli_repaint_output,
    mix_gli_repaint_state,
    validate_gli_repaint_masks,
)
from inference.gli_utils import (  # noqa: E402
    dhw_to_xyz,
    load_cluster_centers,
    nearest_cluster_condition,
    xyz_to_dhw,
)


ASSET_SCRIPT = ROOT / "scripts" / "gli_build_inference_assets.py"
SPEC = importlib.util.spec_from_file_location("gli_build_inference_assets", ASSET_SCRIPT)
ASSETS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ASSETS)

AUDIT_SCRIPT = ROOT / "scripts" / "gli_brain_support_audit.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("gli_brain_support_audit", AUDIT_SCRIPT)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(AUDIT)


MANIFEST_FIELDS = (
    "relative_path",
    "case_id",
    "subject_id",
    "split",
    "patch_size_xyz",
    "anchor_label",
    "anchor_name",
    "sample_role",
    "origin_x",
    "origin_y",
    "origin_z",
    "pad_before_x",
    "pad_before_y",
    "pad_before_z",
    "pad_after_x",
    "pad_after_y",
    "pad_after_z",
    "normalization_p005",
    "normalization_p995",
    "normalization_degenerate",
)


class _CaptureDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.last_x = None

    def forward_with_cond_scale(self, x, time, cond=None, cond_scale=1.0):
        self.last_x = x.detach().clone()
        return torch.zeros_like(x) + self.anchor * 0


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GLIInferenceClosedLoopTests(unittest.TestCase):
    def test_split_v2_preserves_train_and_creates_exact_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            original = {
                **{f"train-{index:03d}": "train" for index in range(584)},
                **{f"holdout-{index:03d}": "val" for index in range(147)},
            }
            source = root / "splits.json"
            source.write_text(json.dumps({"subject_split": original}), encoding="utf-8")
            rows = []
            for subject, split in original.items():
                rows.append(
                    {
                        field: (
                            subject
                            if field in {"case_id", "subject_id"}
                            else split
                            if field == "split"
                            else "1"
                            if field == "anchor_label"
                            else "64x64x32"
                            if field == "patch_size_xyz"
                            else f"patches/{subject}.npz"
                            if field == "relative_path"
                            else "0"
                        )
                        for field in MANIFEST_FIELDS
                    }
                )
            manifest = root / "manifest.csv"
            _write_manifest(manifest, rows)
            output = root / "splits_v2.json"
            payload = ASSETS.build_split_v2(source, manifest, output, seed=20260805)
            counts = payload["subject_counts"]
            self.assertEqual(counts, {"train": 584, "val": 73, "test": 74})
            self.assertTrue(all(payload["subject_split"][key] == "train" for key in original if key.startswith("train")))

    def test_support_mask_and_patch_extraction(self) -> None:
        modalities = []
        for index in range(4):
            array = np.zeros((7, 8, 6), dtype=np.float32)
            array[1:6, 1:7, 1:5] = index + 1
            modalities.append(array)
        modalities[0][0, 0, 0] = 99
        seg = np.zeros_like(modalities[0], dtype=np.uint8)
        seg[3, 3, 3] = 2
        support, outside = build_explicit_brain_support(modalities, seg)
        self.assertFalse(bool(support[0, 0, 0]))
        self.assertTrue(bool(support[3, 3, 3]))
        self.assertEqual(outside, 0)
        patch = extract_xyz_patch(support, (-1, 1, 1), (4, 4, 3), fill_value=False)
        self.assertEqual(patch.shape, (4, 4, 3))
        self.assertFalse(bool(patch[0].any()))
        agreement = AUDIT.support_metrics(support, support)
        self.assertEqual(agreement["dice"], 1.0)
        self.assertEqual(agreement["extra_fraction"], 0.0)

    def test_inference_adapter_scalar_onehot_and_support_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            patch_root = root / "patches" / "patch_4x5x3"
            (patch_root / "patches").mkdir(parents=True)
            t1c = np.zeros((4, 5, 3), dtype=np.float32)
            t1c[1:3, 1:4, :] = 0.5
            seg = np.zeros((4, 5, 3), dtype=np.uint8)
            seg[2, 2, 1] = 3
            hist = np.zeros((4, 16), dtype=np.float32)
            hist[2, 8] = 1
            np.savez_compressed(
                patch_root / "patches/sample.npz",
                t1c=t1c,
                seg=seg,
                hist=hist,
                affine=np.eye(4),
            )
            row = {
                "relative_path": "patches/sample.npz",
                "case_id": "case-1",
                "subject_id": "subject-1",
                "split": "val",
                "patch_size_xyz": "4x5x3",
                "anchor_label": "3",
                "anchor_name": "et",
                "sample_role": "interior",
                "origin_x": "1",
                "origin_y": "1",
                "origin_z": "1",
                "pad_before_x": "0",
                "pad_before_y": "0",
                "pad_before_z": "0",
                "pad_after_x": "0",
                "pad_after_y": "0",
                "pad_after_z": "0",
                "normalization_p005": "0",
                "normalization_p995": "1",
                "normalization_degenerate": "0",
            }
            _write_manifest(patch_root / "manifest.csv", [row])
            split_file = root / "splits_v2.json"
            split_file.write_text(json.dumps({"subject_split": {"subject-1": "test"}}), encoding="utf-8")
            case_dir = root / "raw" / "train" / "case-1"
            case_dir.mkdir(parents=True)
            raw_seg = np.zeros((8, 9, 7), dtype=np.uint8)
            raw_seg[3, 3, 2] = 3
            for modality in ("t1c", "t1n", "t2f", "t2w"):
                image = np.zeros(raw_seg.shape, dtype=np.float32)
                image[1:7, 1:8, 1:6] = 1
                nib.save(nib.Nifti1Image(image, np.eye(4)), case_dir / f"case-1-{modality}.nii.gz")
            nib.save(nib.Nifti1Image(raw_seg, np.eye(4)), case_dir / "case-1-seg.nii.gz")
            sample = GLIInferenceDataset(
                root / "patches", root / "raw", (4, 5, 3), split="test", split_file=split_file
            )[0]
            self.assertEqual(tuple(sample["conditioning_seg"].shape), (1, 3, 4, 5))
            self.assertEqual(tuple(sample["lesion_mask"].shape), (4, 3, 4, 5))
            self.assertTrue(bool(sample["lesion_mask"][2].any()))
            self.assertFalse(bool((sample["healthy_brain_mask"] & sample["outside_mask"]).any()))

    def test_cluster_condition_absent_blocks_and_axis_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "centers.json"
            payload = {
                "schema_version": 1,
                "source_split": "train",
                "patch_size_xyz": [4, 5, 3],
                "condition_dim": 64,
                "labels": [
                    {"value": value, "centers": [np.eye(1, 16, value - 1, dtype=float)[0].tolist()]}
                    for value in (1, 2, 3, 4)
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            centers = load_cluster_centers(path, (4, 5, 3))
            hist = torch.zeros((1, 64))
            hist[0, 32 + 2] = 1
            seg = torch.zeros((1, 1, 3, 4, 5), dtype=torch.long)
            seg[0, 0, 1, 2, 3] = 3
            condition, ids = nearest_cluster_condition(hist, seg, centers)
            self.assertEqual(ids.tolist(), [[-1, -1, 0, -1]])
            self.assertEqual(float(condition[:, :32].sum()), 0.0)
            array = np.arange(60).reshape(3, 4, 5)
            np.testing.assert_array_equal(xyz_to_dhw(dhw_to_xyz(array)), array)

    def test_shared_background_repaint_and_terminal_composition(self) -> None:
        denoiser = _CaptureDenoiser()
        diffusion = GaussianDiffusion_Nolatent(
            denoiser,
            image_size=4,
            num_frames=2,
            spatial_shape=(2, 3, 4),
            channels=4,
            timesteps=4,
            loss_type="l1",
            data_type="gli",
        )
        scalar = torch.zeros((1, 1, 2, 3, 4), dtype=torch.long)
        scalar[0, 0, 0, 0, 0] = 1
        scalar[0, 0, 1, 2, 3] = 4
        lesion_mask = torch.cat([(scalar == value) for value in (1, 2, 3, 4)], dim=1)
        validate_gli_repaint_masks(scalar, lesion_mask, (1, 4, 2, 3, 4))
        generated = torch.arange(4.0).view(1, 4, 1, 1, 1).expand(1, 4, 2, 3, 4)
        background = torch.full((1, 1, 2, 3, 4), 10.0)
        mixed = mix_gli_repaint_state(generated, background, lesion_mask)
        self.assertTrue(torch.equal(mixed[:, :, 0, 1, 1], torch.full((1, 4), 10.0)))
        terminal = mix_gli_repaint_state(generated, background, lesion_mask)
        composed = compose_gli_repaint_output(terminal, lesion_mask)
        self.assertEqual(float(composed[0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(composed[0, 0, 1, 2, 3]), 3.0)
        self.assertEqual(float(composed[0, 0, 0, 1, 1]), 10.0)
        sampled = diffusion.p_sample_repaint(
            generated.clone(),
            torch.tensor([0]),
            cond=torch.zeros((1, 64)),
            conf=SimpleNamespace(inpa_inj_sched_prev_cumnoise=False),
            model_kwargs={
                "gt": background.expand_as(generated),
                "gt_background": background,
                "gt_keep_mask": scalar,
                "lesion_mask": lesion_mask,
                "background_noise": torch.zeros_like(background),
            },
        )
        captured = denoiser.last_x
        self.assertTrue(torch.equal(captured[:, 0, 0, 1, 1], captured[:, 3, 0, 1, 1]))
        self.assertTrue(torch.equal(sampled[:, :, 0, 1, 1], torch.full((1, 4), 10.0)))

    def test_versioned_checkpoint_metadata_and_hydra_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = torch.nn.Linear(2, 2)
            expected = {
                "data_type": "gli",
                "diffusion_num_channels": 4,
                "cond_dim": 64,
                "base_dim": 64,
                "spatial_shape_dhw": [2, 3, 4],
                "timesteps": 4,
                "temporal_max_distance": 128,
            }
            path = Path(temporary_dir) / "checkpoint.pt"
            torch.save(
                {"schema_version": 1, "metadata": expected, "model": model.state_dict(), "ema": model.state_dict()},
                path,
            )
            load_diffusion_checkpoint(model, path, weights_key="model", expected_metadata=expected)
            wrong = dict(expected)
            wrong["cond_dim"] = 32
            with self.assertRaisesRegex(ValueError, "cond_dim"):
                load_diffusion_checkpoint(model, path, weights_key="model", expected_metadata=wrong)

        for name, expected_shape in (
            ("gli_64x64x32.yaml", [32, 64, 64]),
            ("gli_80x96x80.yaml", [80, 80, 96]),
        ):
            cfg = OmegaConf.load(ROOT / "LeFusion" / "inference" / "confs" / name)
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            self.assertEqual(list(cfg.model.spatial_shape_dhw), expected_shape)


if __name__ == "__main__":
    unittest.main()

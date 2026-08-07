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
    anchor_union_cluster_condition,
    dhw_to_xyz,
    indexed_cluster_condition,
    load_cluster_centers,
    mask_input_inside_lesion,
    nearest_cluster_condition,
    resolve_union_target_labels,
    union_label_cluster_condition,
    xyz_to_dhw,
)
from inference.gli_selection import (  # noqa: E402
    build_selection_manifest,
    manifest_shard_paths,
    select_stratified_fraction,
    select_val_qa,
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

INFERENCE_SCRIPT = ROOT / "LeFusion" / "inference" / "inference.py"
INFERENCE_SPEC = importlib.util.spec_from_file_location("gli_inference_entrypoint", INFERENCE_SCRIPT)
INFERENCE = importlib.util.module_from_spec(INFERENCE_SPEC)
assert INFERENCE_SPEC.loader is not None
INFERENCE_SPEC.loader.exec_module(INFERENCE)


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
    def test_direct_inference_entrypoint_imports_fixed_union_resolver(self) -> None:
        self.assertIs(INFERENCE.resolve_union_target_labels, resolve_union_target_labels)

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

    def test_masked_input_and_anchor_union_condition(self) -> None:
        input_t1c = torch.arange(24, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
        seg = torch.zeros((1, 1, 2, 3, 4), dtype=torch.long)
        seg[0, 0, 0, 0, 0] = 1
        seg[0, 0, 1, 2, 3] = 4
        masked = mask_input_inside_lesion(input_t1c, seg, fill_value=0.0)
        self.assertEqual(float(masked[0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(masked[0, 0, 1, 2, 3]), 0.0)
        self.assertTrue(torch.equal(masked[seg == 0], input_t1c[seg == 0]))

        hist = torch.zeros((1, 64), dtype=torch.float32)
        hist[0, 48 + 5] = 1.0
        centers = [torch.eye(16, dtype=torch.float32)[:2] for _ in range(4)]
        centers[3] = torch.stack(
            [torch.eye(16, dtype=torch.float32)[2], torch.eye(16, dtype=torch.float32)[5]]
        )
        target_seg, target_mask, condition, cluster_ids = anchor_union_cluster_condition(
            hist, seg, torch.tensor([4]), centers
        )
        self.assertEqual(set(torch.unique(target_seg).tolist()), {0, 4})
        self.assertEqual(target_mask.reshape(1, 4, -1).sum(dim=2).tolist(), [[0.0, 0.0, 0.0, 2.0]])
        self.assertEqual(cluster_ids.tolist(), [[-1, -1, -1, 1]])
        self.assertEqual(float(condition[:, :48].abs().sum()), 0.0)
        self.assertEqual(float(condition[:, 48:].sum()), 1.0)

        indexed, indexed_ids = indexed_cluster_condition(
            hist, seg, centers, index_mode="last"
        )
        self.assertEqual(indexed_ids.tolist(), [[1, -1, -1, 1]])
        self.assertEqual(float(indexed[:, 16:48].sum()), 0.0)

        cycle_seg, cycle_mask, cycle_condition, cycle_ids = union_label_cluster_condition(
            hist,
            seg,
            torch.tensor([2]),
            centers,
            index_mode="first",
        )
        self.assertEqual(set(torch.unique(cycle_seg).tolist()), {0, 2})
        self.assertEqual(cycle_mask.reshape(1, 4, -1).sum(dim=2).tolist(), [[0.0, 2.0, 0.0, 0.0]])
        self.assertEqual(cycle_ids.tolist(), [[-1, 0, -1, -1]])
        self.assertEqual(float(cycle_condition[:, :16].sum()), 0.0)
        self.assertEqual(float(cycle_condition[:, 16:32].sum()), 1.0)

        anchor_labels = torch.tensor([4, 1], dtype=torch.long)
        cycled = resolve_union_target_labels(
            anchor_labels, lesion_mode="union_single_label_cycle", batch_index=5
        )
        fixed = resolve_union_target_labels(
            anchor_labels,
            lesion_mode="union_single_label_fixed",
            batch_index=5,
            fixed_target_label=3,
        )
        self.assertEqual(cycled.tolist(), [2, 2])
        self.assertEqual(fixed.tolist(), [3, 3])
        with self.assertRaisesRegex(ValueError, "fixed_target_label"):
            resolve_union_target_labels(
                anchor_labels,
                lesion_mode="union_single_label_fixed",
                batch_index=0,
                fixed_target_label=5,
            )

    def test_single_state_repaint_uses_union_and_restores_exact_background(self) -> None:
        scalar = torch.zeros((1, 1, 1, 2, 3), dtype=torch.long)
        scalar[0, 0, 0, 0, 0] = 1
        scalar[0, 0, 0, 1, 2] = 4
        lesion_mask = torch.cat([(scalar == value) for value in (1, 2, 3, 4)], dim=1)
        validate_gli_repaint_masks(scalar, lesion_mask, (1, 1, 1, 2, 3))
        generated = torch.full((1, 1, 1, 2, 3), 2.0)
        background = torch.full_like(generated, -0.25)
        mixed = mix_gli_repaint_state(generated, background, lesion_mask)
        self.assertEqual(float(mixed[0, 0, 0, 0, 0]), 2.0)
        self.assertEqual(float(mixed[0, 0, 0, 0, 1]), -0.25)
        composed = compose_gli_repaint_output(
            mixed, lesion_mask, background_context=background
        )
        union = lesion_mask.any(dim=1, keepdim=True)
        self.assertTrue(torch.equal(composed[~union], background[~union]))

    def test_pre_denoiser_shared_background_uses_fresh_noise_and_composition(self) -> None:
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
        kwargs = {
            "gt": background.expand_as(generated),
            "gt_background": background,
            "gt_keep_mask": scalar,
            "lesion_mask": lesion_mask,
        }
        torch.manual_seed(17)
        diffusion.p_sample_repaint(
            generated.clone(),
            torch.tensor([1]),
            cond=torch.zeros((1, 64)),
            conf=SimpleNamespace(inpa_inj_sched_prev_cumnoise=False),
            model_kwargs=kwargs,
        )
        first_captured = denoiser.last_x.clone()
        diffusion.p_sample_repaint(
            generated.clone(),
            torch.tensor([1]),
            cond=torch.zeros((1, 64)),
            conf=SimpleNamespace(inpa_inj_sched_prev_cumnoise=False),
            model_kwargs=kwargs,
        )
        second_captured = denoiser.last_x.clone()
        self.assertTrue(torch.equal(first_captured[:, 0, 0, 1, 1], first_captured[:, 3, 0, 1, 1]))
        self.assertTrue(torch.equal(second_captured[:, 0, 0, 1, 1], second_captured[:, 3, 0, 1, 1]))
        self.assertFalse(torch.equal(first_captured[:, 0, 0, 1, 1], second_captured[:, 0, 0, 1, 1]))
        self.assertNotIn("background_noise", kwargs)

    def test_post_denoiser_state_is_not_hard_clamped(self) -> None:
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
        lesion_mask = torch.cat([(scalar == value) for value in (1, 2, 3, 4)], dim=1)
        generated = torch.zeros((1, 4, 2, 3, 4))
        background = torch.full((1, 1, 2, 3, 4), 10.0)
        model_mean = torch.arange(1.0, 5.0).view(1, 4, 1, 1, 1).expand_as(generated)

        def fixed_mean(**_kwargs):
            return model_mean, torch.zeros_like(model_mean), torch.zeros_like(model_mean)

        diffusion.p_mean_variance = fixed_mean
        sampled = diffusion.p_sample_repaint(
            generated,
            torch.tensor([0]),
            cond=torch.zeros((1, 64)),
            conf=SimpleNamespace(inpa_inj_sched_prev_cumnoise=False),
            model_kwargs={
                "gt": background.expand_as(generated),
                "gt_background": background,
                "gt_keep_mask": scalar,
                "lesion_mask": lesion_mask,
            },
        )
        self.assertTrue(torch.equal(sampled, model_mean))
        self.assertFalse(torch.equal(sampled[:, :, 0, 1, 1], background.expand_as(sampled)[:, :, 0, 1, 1]))

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

            formal_path = Path(temporary_dir) / "formal.pt"
            torch.save(
                {
                    "schema_version": 2,
                    "metadata": {"data_type": "gli"},
                    "resolved_config": {"model": expected},
                    "model": model.state_dict(),
                    "ema": model.state_dict(),
                },
                formal_path,
            )
            loaded = load_diffusion_checkpoint(
                model, formal_path, weights_key="ema", expected_metadata=expected
            )
            self.assertEqual(loaded["schema_version"], 2)

        for name, expected_shape in (
            ("gli_64x64x32.yaml", [32, 64, 64]),
            ("gli_80x96x80.yaml", [80, 80, 96]),
            ("gli_exp005_p64_val_qa.yaml", [32, 64, 64]),
            ("gli_exp005_p64_test_subset50_shard0.yaml", [32, 64, 64]),
            ("gli_exp005_p64_test_subset50_shard1.yaml", [32, 64, 64]),
            ("gli_exp006_p64_val_qa.yaml", [32, 64, 64]),
            ("gli_exp006_p64_test_subset50_shard0.yaml", [32, 64, 64]),
            ("gli_exp006_p64_test_subset50_shard1.yaml", [32, 64, 64]),
            ("gli_exp007_p64_val_qa_masked_multilabel.yaml", [32, 64, 64]),
            ("gli_exp007_p64_val_qa_masked_anchor_union.yaml", [32, 64, 64]),
            ("gli_exp008_p64_val_qa_masked_multilabel.yaml", [32, 64, 64]),
            ("gli_exp008_p64_val_qa_masked_anchor_union.yaml", [32, 64, 64]),
        ):
            cfg = OmegaConf.load(ROOT / "LeFusion" / "inference" / "confs" / name)
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            self.assertEqual(list(cfg.model.spatial_shape_dhw), expected_shape)
            if name.startswith(("gli_exp005", "gli_exp006")):
                self.assertEqual(str(cfg.checkpoint.weights_key), "ema")
                self.assertEqual(int(cfg.repaint.schedule_jump_params.t_T), 300)
            if name.startswith("gli_exp006"):
                self.assertEqual(str(cfg.repaint.background_noise), "fresh_per_reverse_call_shared_across_channels")
                self.assertFalse(bool(cfg.repaint.post_denoiser_hard_clamp))
            if name.startswith("gli_exp007"):
                self.assertEqual(str(cfg.checkpoint.weights_key), "ema")
                self.assertEqual(int(cfg.repaint.schedule_jump_params.t_T), 300)
                self.assertTrue(bool(cfg.input_policy.mask_inside_lesion))
                self.assertEqual(float(cfg.input_policy.fill_value), 0.0)
            if name.startswith("gli_exp008"):
                self.assertEqual(str(cfg.checkpoint.weights_key), "ema")
                self.assertEqual(int(cfg.model.spatial_condition_channels), 5)
                self.assertEqual(int(cfg.repaint.schedule_jump_params.t_T), 300)
                self.assertTrue(bool(cfg.input_policy.mask_inside_lesion))
                self.assertEqual(float(cfg.input_policy.fill_value), 0.0)

    def test_deterministic_half_selection_and_shards(self) -> None:
        records = []
        for label in (1, 2, 3, 4):
            for role in ("interior", "boundary"):
                for index in range(5):
                    records.append(
                        {
                            "relative_path": f"{label}/{role}/{index}.npz",
                            "case_id": f"case-{label}-{role}-{index}",
                            "subject_id": f"subject-{label}-{index}",
                            "anchor_label": str(label),
                            "sample_role": role,
                        }
                    )
        selected, selection = select_stratified_fraction(records, fraction=0.5, seed=20260806)
        repeated, _ = select_stratified_fraction(records, fraction=0.5, seed=20260806)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 20)
        self.assertEqual(set(selection["stratum_quotas"].values()), {2, 3})
        payload = build_selection_manifest(
            records,
            selected,
            selection=selection,
            split="test",
            patch_size_xyz=(64, 64, 32),
            shard_count=2,
            provenance={"dataset_manifest_sha256": "a", "split_sha256": "b"},
        )
        shard0 = manifest_shard_paths(payload, shard_index=0, shard_count=2)
        shard1 = manifest_shard_paths(payload, shard_index=1, shard_count=2)
        self.assertFalse(set(shard0).intersection(shard1))
        self.assertEqual(set(shard0).union(shard1), set(payload["selected_relative_paths"]))

    def test_val_qa_selection_covers_label_role_and_prefers_multilabel(self) -> None:
        records = []
        counts = {}
        for label in (1, 2, 3, 4):
            for role in ("interior", "boundary"):
                for index in range(2):
                    relative = f"{label}/{role}/{index}.npz"
                    records.append(
                        {
                            "relative_path": relative,
                            "case_id": relative,
                            "subject_id": f"subject-{label}-{index}",
                            "anchor_label": str(label),
                            "sample_role": role,
                        }
                    )
                    counts[relative] = index + 1
        selected, _ = select_val_qa(records, seed=20260806, lesion_label_counts=counts)
        self.assertEqual(len(selected), 8)
        chosen = [records[index] for index in selected]
        self.assertEqual(
            {(int(row["anchor_label"]), row["sample_role"]) for row in chosen},
            {(label, role) for label in (1, 2, 3, 4) for role in ("interior", "boundary")},
        )
        self.assertTrue(all(counts[row["relative_path"]] == 2 for row in chosen))


if __name__ == "__main__":
    unittest.main()

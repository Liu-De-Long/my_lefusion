from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from LeFusion.classifier.engine import (
    _cyclic_epoch_records,
    _feature_rows_and_target,
    _select_balanced_pseudo_voxels,
    evaluate_model,
    update_selection_state,
    validate_test_gate,
)
from LeFusion.classifier.data import (
    GLIClassifierPatchDataset,
    SAFE_SAMPLE_KEYS,
    build_labeled_subset,
)
from LeFusion.classifier.features import FEATURE_CHANNELS, build_feature_volume
from LeFusion.classifier.metrics import PatientMetricAccumulator, reconstruct_prediction
from LeFusion.classifier.models import build_classifier, count_parameters
from LeFusion.classifier.tracking import validate_classifier_wandb_config


MANIFEST_FIELDS = [
    "relative_path",
    "case_id",
    "subject_id",
    "patch_size_xyz",
    "anchor_label",
    "sample_role",
    "label_1_voxels",
    "label_2_voxels",
    "label_3_voxels",
    "label_4_voxels",
]


class TestGLIClassifierData(unittest.TestCase):
    def test_subset_is_exactly_balanced_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            size_root = root / "patch_64x64x32"
            size_root.mkdir()
            rows = []
            subject_split = {"val-subject": "val"}
            for label in range(1, 5):
                for role in ("interior", "boundary"):
                    for index in range(4):
                        subject = f"subject-{label}-{role}-{index % 3}"
                        subject_split[subject] = "train"
                        rows.append(
                            {
                                "relative_path": f"patches/{label}_{role}_{index}.npz",
                                "case_id": f"case-{label}-{role}-{index}",
                                "subject_id": subject,
                                "patch_size_xyz": "64x64x32",
                                "anchor_label": str(label),
                                "sample_role": role,
                                "label_1_voxels": "10",
                                "label_2_voxels": "20",
                                "label_3_voxels": "30",
                                "label_4_voxels": "40",
                            }
                        )
            with (size_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            split_file = root / "splits.json"
            split_file.write_text(
                json.dumps({"subject_split": subject_split}), encoding="utf-8"
            )
            output = root / "subset.json"
            payload = build_labeled_subset(root, split_file, output, count=16, seed=7)
            self.assertEqual(payload["actual_count"], 16)
            self.assertEqual(payload["anchor_counts"], {"1": 4, "2": 4, "3": 4, "4": 4})
            self.assertEqual(payload["role_counts"], {"boundary": 8, "interior": 8})
            self.assertEqual(set(payload["stratum_counts"].values()), {2})

    def test_dataset_transposes_xyz_and_never_returns_hist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            size_root = root / "patch_64x64x32" / "patches"
            size_root.mkdir(parents=True)
            t1c = np.zeros((64, 64, 32), dtype=np.float32)
            seg = np.zeros((64, 64, 32), dtype=np.uint8)
            t1c[2, 3, 4] = 0.75
            seg[2, 3, 4] = 3
            np.savez_compressed(
                size_root / "sample.npz",
                t1c=t1c,
                seg=seg,
                hist=np.zeros((4, 16), dtype=np.float32),
                affine=np.eye(4, dtype=np.float32),
            )
            permuted_seg = seg.copy()
            permuted_seg[permuted_seg == 3] = 4
            np.savez_compressed(
                size_root / "permuted.npz",
                t1c=t1c,
                seg=permuted_seg,
                hist=np.ones((4, 16), dtype=np.float32),
                affine=np.eye(4, dtype=np.float32),
            )
            dataset = GLIClassifierPatchDataset(
                root,
                [
                    {
                        "relative_path": "patches/sample.npz",
                        "case_id": "case",
                        "subject_id": "subject",
                    },
                    {
                        "relative_path": "patches/permuted.npz",
                        "case_id": "case-permuted",
                        "subject_id": "subject-permuted",
                    },
                ],
            )
            sample = dataset[0]
            permuted = dataset[1]
            self.assertTrue(set(sample).issubset(SAFE_SAMPLE_KEYS))
            self.assertNotIn("hist", sample)
            self.assertNotIn("anchor_label", sample)
            self.assertEqual(tuple(sample["image"].shape), (1, 32, 64, 64))
            self.assertEqual(float(sample["image"][0, 4, 2, 3]), 0.75)
            self.assertEqual(int(sample["target"][4, 2, 3]), 3)
            self.assertTrue(torch.equal(sample["image"], permuted["image"]))
            self.assertTrue(torch.equal(sample["total_mask"], permuted["total_mask"]))
            self.assertFalse(torch.equal(sample["target"], permuted["target"]))
            unlabeled_dataset = GLIClassifierPatchDataset(
                root,
                [
                    {
                        "relative_path": "patches/sample.npz",
                        "case_id": "case",
                        "subject_id": "subject",
                    }
                ],
                load_targets=False,
            )
            self.assertNotIn("target", unlabeled_dataset[0])


class TestGLIClassifierFeaturesAndModels(unittest.TestCase):
    def test_features_depend_only_on_image_and_union(self) -> None:
        image = torch.linspace(-1, 1, 4 * 5 * 6).reshape(4, 5, 6)
        mask = torch.zeros((4, 5, 6), dtype=torch.bool)
        mask[1:4, 1:4, 2:6] = True
        for kind, channels in FEATURE_CHANNELS.items():
            first = build_feature_volume(image, mask, kind)
            second = build_feature_volume(image.clone(), mask.clone(), kind)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(tuple(first.shape), (channels, 4, 5, 6))

    def test_model_shapes_and_parameter_budgets(self) -> None:
        for kind in ("m0", "m1", "m2"):
            model = build_classifier(kind)
            output = model(torch.zeros(7, FEATURE_CHANNELS[kind]))
            self.assertEqual(tuple(output.shape), (7, 4))
            self.assertLess(count_parameters(model), 50_000)
        cnn = build_classifier("c0", cnn_channels=24)
        output = cnn(torch.zeros(1, 2, 4, 8, 8))
        self.assertEqual(tuple(output.shape), (1, 4, 4, 8, 8))
        self.assertLess(count_parameters(cnn), 400_000)
        unet = build_classifier("unet3d", unet_base_channels=8)
        output = unet(torch.zeros(1, 2, 8, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 4, 8, 16, 16))
        self.assertLess(count_parameters(unet), 2_000_000)

    def test_feature_row_cache_keeps_targets_separate_and_reuses_rows(self) -> None:
        image = torch.linspace(-1, 1, 4 * 5 * 6).reshape(1, 4, 5, 6)
        mask = torch.zeros((1, 4, 5, 6), dtype=torch.bool)
        mask[:, 1:4, 1:4, 2:6] = True
        target = torch.zeros((4, 5, 6), dtype=torch.int64)
        target[mask[0]] = 3
        sample = {
            "relative_path": "patches/frozen.npz",
            "image": image,
            "total_mask": mask,
            "target": target,
        }
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        rows, target_rows = _feature_rows_and_target(
            sample, feature_kind="m1", cache=cache
        )
        sample["image"] = torch.full_like(image, 99)
        sample["target"] = torch.full_like(target, 4)
        cached_rows, cached_targets = _feature_rows_and_target(
            sample, feature_kind="m1", cache=cache
        )
        self.assertEqual(rows.data_ptr(), cached_rows.data_ptr())
        self.assertEqual(target_rows.data_ptr(), cached_targets.data_ptr())
        self.assertEqual(set(target_rows.tolist()), {2})
        self.assertFalse(torch.any(rows == 99))


class TestGLIClassifierMetrics(unittest.TestCase):
    def test_spatial_batched_evaluation_matches_single_patch_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_root = root / "patch_64x64x32" / "patches"
            patch_root.mkdir(parents=True)
            records = []
            for index in range(4):
                image = np.zeros((64, 64, 32), dtype=np.float32)
                seg = np.zeros((64, 64, 32), dtype=np.uint8)
                seg[16:48, 16:48, 8:24] = (index % 4) + 1
                path = patch_root / f"sample-{index}.npz"
                np.savez_compressed(
                    path,
                    t1c=image,
                    seg=seg,
                    hist=np.zeros((4, 16), dtype=np.float32),
                    affine=np.eye(4, dtype=np.float32),
                )
                records.append(
                    {
                        "relative_path": f"patches/{path.name}",
                        "case_id": f"case-{index}",
                        "subject_id": f"subject-{index}",
                    }
                )
            model = build_classifier("c0", cnn_channels=4).eval()
            single = evaluate_model(
                model,
                kind="c0",
                dataset_root=root,
                records=records,
                device=torch.device("cpu"),
                inference_voxels=1024,
                spatial_batch_size=1,
                bootstrap_samples=0,
                seed=3,
            )
            batched = evaluate_model(
                model,
                kind="c0",
                dataset_root=root,
                records=records,
                device=torch.device("cpu"),
                inference_voxels=1024,
                spatial_batch_size=2,
                bootstrap_samples=0,
                seed=3,
            )
            self.assertTrue(
                np.allclose(
                    single["patient_normalized_confusion_matrix"],
                    batched["patient_normalized_confusion_matrix"],
                )
            )
            self.assertAlmostEqual(single["focus_miou"], batched["focus_miou"])

    def test_unlabeled_epoch_sampler_covers_pool_without_label_metadata(self) -> None:
        records = [
            {
                "relative_path": f"patch-{index}.npz",
                "case_id": f"case-{index}",
                "subject_id": f"subject-{index}",
            }
            for index in range(7)
        ]
        covered = set()
        for epoch in range(3):
            sampled = _cyclic_epoch_records(records, count=3, seed=11, epoch=epoch)
            covered.update(item["relative_path"] for item in sampled)
        self.assertEqual(covered, {item["relative_path"] for item in records})

    def test_pseudo_voxel_selection_is_confident_and_class_balanced(self) -> None:
        confidence = torch.tensor([[[0.99, 0.98, 0.97, 0.96, 0.95, 0.70]]])
        pseudo = torch.tensor([[[0, 0, 0, 1, 1, 2]]])
        mask = torch.ones_like(pseudo, dtype=torch.bool)
        selected = _select_balanced_pseudo_voxels(
            confidence,
            pseudo,
            mask,
            threshold=0.9,
            max_per_class=2,
        )
        self.assertEqual(int(selected.sum()), 4)
        self.assertTrue(torch.all(confidence[selected] >= 0.9))
        self.assertEqual(int((pseudo[selected] == 0).sum()), 2)
        self.assertEqual(int((pseudo[selected] == 1).sum()), 2)

    def test_formal_wandb_config_is_online_and_fail_closed(self) -> None:
        config = {
            "wandb": {
                "enabled": True,
                "fail_closed": True,
                "mode": "online",
                "entity": "entity",
                "project": "project",
                "run_id": "run-id",
                "run_name": "run-name",
                "dir": "outputs/wandb",
            }
        }
        self.assertEqual(
            validate_classifier_wandb_config(config, resume=False)["resume"], "never"
        )
        self.assertEqual(
            validate_classifier_wandb_config(config, resume=True)["resume"], "must"
        )
        config["wandb"]["mode"] = "offline"
        with self.assertRaises(ValueError):
            validate_classifier_wandb_config(config, resume=False)

    def test_absolute_best_is_independent_from_early_stopping_delta(self) -> None:
        state = update_selection_state(
            0.5436,
            best_metric=0.5436,
            patience_metric=0.5436,
            bad_epochs=2,
            min_delta=0.005,
        )
        best, patience_reference, bad_epochs, is_best, is_significant = state
        self.assertEqual(best, 0.5436)
        self.assertEqual(patience_reference, 0.5436)
        self.assertEqual(bad_epochs, 3)
        self.assertFalse(is_best)
        self.assertFalse(is_significant)

        state = update_selection_state(
            0.5482,
            best_metric=best,
            patience_metric=patience_reference,
            bad_epochs=bad_epochs,
            min_delta=0.005,
        )
        best, patience_reference, bad_epochs, is_best, is_significant = state
        self.assertEqual(best, 0.5482)
        self.assertEqual(patience_reference, 0.5436)
        self.assertEqual(bad_epochs, 4)
        self.assertTrue(is_best)
        self.assertFalse(is_significant)

    def test_reconstruction_forces_background_and_patient_metrics(self) -> None:
        mask = torch.zeros((2, 2, 2), dtype=torch.bool)
        mask.flatten()[:4] = True
        prediction = reconstruct_prediction(torch.tensor([0, 1, 2, 3]), mask)
        self.assertTrue(torch.equal(prediction[~mask], torch.zeros(4, dtype=torch.int64)))
        self.assertEqual(set(prediction[mask].tolist()), {1, 2, 3, 4})

        target = prediction.clone()
        accumulator = PatientMetricAccumulator()
        accumulator.update(
            subject_id="subject-a",
            prediction=prediction,
            target=target,
            total_mask=mask,
        )
        metrics = accumulator.compute(bootstrap_samples=10, seed=1)
        self.assertEqual(metrics["focus_miou"], 1.0)
        self.assertEqual(metrics["outside_nonzero_rate"], 0.0)
        self.assertEqual(metrics["union_dice"], 1.0)
        self.assertTrue(all(metrics["gate"].values()))
        json.dumps(metrics)

    def test_test_gate_is_bound_to_exact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "best.pt"
            subset = output / "subset.json"
            checkpoint.write_bytes(b"checkpoint")
            subset.write_text("{}", encoding="utf-8")
            payload = {
                "gate": {
                    "focus_miou_at_least_0_85": True,
                    "et_iou_at_least_0_80": True,
                    "rc_iou_at_least_0_80": True,
                    "outside_nonzero_is_zero": True,
                    "union_dice_is_one": True,
                },
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "config_sha256": "config-hash",
                "subset_sha256": hashlib.sha256(b"{}").hexdigest(),
            }
            (output / "best_val_metrics.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            self.assertEqual(
                validate_test_gate(
                    checkpoint,
                    config_sha256="config-hash",
                    subset_path=subset,
                )["checkpoint_sha256"],
                payload["checkpoint_sha256"],
            )
            payload["gate"]["focus_miou_at_least_0_85"] = False
            (output / "best_val_metrics.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                validate_test_gate(
                    checkpoint,
                    config_sha256="config-hash",
                    subset_path=subset,
                )


if __name__ == "__main__":
    unittest.main()

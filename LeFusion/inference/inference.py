import os
import io
import blobfile as bf
import torch as th
import json
import sys
import csv
import time
from pathlib import Path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import GaussianDiffusion_Nolatent, Unet3D, normalize_spatial_shape
from get_dataset.get_dataset import get_inference_dataloader
from train.train import build_model_and_diffusion
from checkpointing import load_diffusion_checkpoint, sha256_file
if __package__:
    from inference.gli_utils import (
        anchor_union_cluster_condition,
        dhw_to_xyz,
        indexed_cluster_condition,
        load_cluster_centers,
        mask_input_inside_lesion,
        nearest_cluster_condition,
        union_label_cluster_condition,
    )
    from inference.gli_selection import manifest_shard_paths
else:
    from gli_utils import (
        anchor_union_cluster_condition,
        dhw_to_xyz,
        indexed_cluster_condition,
        load_cluster_centers,
        mask_input_inside_lesion,
        nearest_cluster_condition,
        union_label_cluster_condition,
    )
    from gli_selection import manifest_shard_paths
import torchio as tio
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra

def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


def perturb_tensor(tensor, mean=0.0, std=1.0, bili=0.1):
    perturbation = th.normal(mean, std, size=tensor.size(), device=tensor.device)
    perturbation -= perturbation.mean()
    denominator = perturbation.abs().max()
    if not bool(denominator > 0) or not bool(tensor.abs().max() > 0):
        return tensor.clone()
    max_perturbation = tensor.abs() * bili
    perturbation = perturbation / denominator * max_perturbation
    perturbed_tensor = tensor + perturbation
    return perturbed_tensor


def _gli_expected_metadata(conf: DictConfig) -> dict:
    base_dim = conf.model.get('base_dim') or conf.model.diffusion_img_size
    return {
        "data_type": "gli",
        "diffusion_num_channels": int(conf.model.diffusion_num_channels),
        "cond_dim": int(conf.model.cond_dim),
        "base_dim": int(base_dim),
        "spatial_shape_dhw": [int(value) for value in conf.model.spatial_shape_dhw],
        "timesteps": int(conf.model.timesteps),
        "temporal_max_distance": int(conf.model.temporal_max_distance),
        "spatial_condition_channels": int(
            conf.model.get("spatial_condition_channels", 0)
        ),
        "objective": str(
            conf.get("lesion_generation", {}).get("objective", "pred_noise")
        ),
        "gli_state_mode": str(
            conf.get("lesion_generation", {}).get("state_mode", "full_t1c")
        ),
    }


def _safe_output_root(path: str, overwrite: bool, resume: bool = False) -> Path:
    output = Path(path).expanduser()
    if output.exists() and any(output.iterdir()) and not overwrite and not resume:
        raise FileExistsError(f"GLI inference output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("generated_npz", "generated_nifti", "qa"):
        (output / name).mkdir(exist_ok=True)
    return output


def _checkpoint_git_sha(checkpoint: dict) -> str:
    metadata = checkpoint.get("metadata", {})
    return str(metadata.get("git_sha", metadata.get("git_commit", "unknown")))


def _sampling_seed(base_seed: int, relative_path: str) -> int:
    import hashlib

    digest = hashlib.sha256(
        f"{int(base_seed)}\0{relative_path}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _load_progress(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "manifest" not in record or "metrics" not in record:
                raise ValueError(f"invalid progress record at line {line_number}: {path}")
            records.append(record)
    paths = [str(record["manifest"]["source_relative_path"]) for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate completed path in progress file: {path}")
    return records


def _append_progress(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def _boundary_edge_jumps(
    image_dhw: np.ndarray,
    lesion_dhw: np.ndarray,
    support_dhw: np.ndarray,
) -> np.ndarray:
    """Return absolute lesion/healthy-support jumps across 6-neighbour edges."""
    healthy = support_dhw & ~lesion_dhw
    values = []
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        lower = tuple(lower)
        upper = tuple(upper)
        crossing = (
            (lesion_dhw[lower] & healthy[upper])
            | (healthy[lower] & lesion_dhw[upper])
        )
        if crossing.any():
            values.append(np.abs(image_dhw[lower] - image_dhw[upper])[crossing])
    if not values:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


def _region_metrics(
    generated_dhw: np.ndarray,
    input_dhw: np.ndarray,
    seg_dhw: np.ndarray,
    support_dhw: np.ndarray,
    condition: np.ndarray,
) -> dict:
    lesion = seg_dhw > 0
    healthy = support_dhw & ~lesion
    outside = ~support_dhw
    difference = np.abs(generated_dhw - input_dhw)
    healthy_values = difference[healthy]
    outside_values = np.abs(generated_dhw[outside])
    outside_changes = difference[outside]
    outside_input_values = np.abs(input_dhw[outside])
    lesion_values = difference[lesion]
    generated_lesion = generated_dhw[lesion]
    input_lesion = input_dhw[lesion]
    background_values = difference[~lesion]
    zero_fill_baseline_mae = (
        float(np.abs(input_lesion).mean()) if input_lesion.size else None
    )
    lesion_mae = float(lesion_values.mean()) if lesion_values.size else None
    shell = ndimage.binary_dilation(lesion, iterations=1) & ~lesion & support_dhw
    shell_values = difference[shell]
    input_boundary_jumps = _boundary_edge_jumps(input_dhw, lesion, support_dhw)
    generated_boundary_jumps = _boundary_edge_jumps(generated_dhw, lesion, support_dhw)
    input_boundary_p95 = _percentile(input_boundary_jumps, 95)
    generated_boundary_p95 = _percentile(generated_boundary_jumps, 95)
    metrics = {
        "healthy_brain_mae": float(healthy_values.mean()) if healthy_values.size else None,
        "healthy_brain_p95_abs_change": _percentile(healthy_values, 95),
        "healthy_brain_max_abs": float(healthy_values.max()) if healthy_values.size else None,
        "healthy_brain_changed_fraction_gt_0p1": (
            float(np.mean(healthy_values > 0.1)) if healthy_values.size else None
        ),
        # Retained only for exp005 hard-clamp comparison; no longer a QA gate.
        "healthy_brain_exact": bool(not healthy_values.size or np.all(healthy_values == 0)),
        "outside_mean_abs": float(outside_values.mean()) if outside_values.size else None,
        "outside_max_abs": float(outside_values.max()) if outside_values.size else None,
        "outside_input_mean_abs": (
            float(outside_input_values.mean()) if outside_input_values.size else None
        ),
        "outside_change_mae": (
            float(outside_changes.mean()) if outside_changes.size else None
        ),
        "outside_change_p95_abs": _percentile(outside_changes, 95),
        "outside_change_max_abs": (
            float(outside_changes.max()) if outside_changes.size else None
        ),
        "outside_changed_fraction_gt_0p1": (
            float(np.mean(outside_changes > 0.1)) if outside_changes.size else None
        ),
        # Retained only for exp005 hard-clamp comparison; no longer a QA gate.
        "outside_change_exact": bool(
            not outside_changes.size or np.all(outside_changes == 0)
        ),
        "outside_nonzero_fraction_input": (
            float(np.mean(outside_input_values > 1e-6)) if outside_input_values.size else None
        ),
        "outside_nonzero_fraction_generated": (
            float(np.mean(outside_values > 1e-6)) if outside_values.size else None
        ),
        "lesion_change_mae": lesion_mae,
        "zero_fill_baseline_lesion_mae": zero_fill_baseline_mae,
        "lesion_mae_improvement_fraction": (
            float((zero_fill_baseline_mae - lesion_mae) / zero_fill_baseline_mae)
            if zero_fill_baseline_mae not in {None, 0.0} and lesion_mae is not None
            else None
        ),
        "generated_lesion_mean_abs": (
            float(np.abs(generated_lesion).mean()) if generated_lesion.size else None
        ),
        "generated_lesion_std": (
            float(generated_lesion.std()) if generated_lesion.size else None
        ),
        "background_max_abs_change": (
            float(background_values.max()) if background_values.size else None
        ),
        "boundary_outer_shell_mae": float(shell_values.mean()) if shell_values.size else None,
        "boundary_outer_shell_p95_abs_change": _percentile(shell_values, 95),
        "boundary_outer_shell_max_abs": (
            float(shell_values.max()) if shell_values.size else None
        ),
        "boundary_outer_shell_changed_fraction_gt_0p1": (
            float(np.mean(shell_values > 0.1)) if shell_values.size else None
        ),
        "boundary_edge_count": int(input_boundary_jumps.size),
        "boundary_jump_input_mean": (
            float(input_boundary_jumps.mean()) if input_boundary_jumps.size else None
        ),
        "boundary_jump_input_p95": input_boundary_p95,
        "boundary_jump_generated_mean": (
            float(generated_boundary_jumps.mean()) if generated_boundary_jumps.size else None
        ),
        "boundary_jump_generated_p95": generated_boundary_p95,
        "boundary_jump_mean_increase": (
            float(generated_boundary_jumps.mean() - input_boundary_jumps.mean())
            if input_boundary_jumps.size else None
        ),
        "boundary_jump_p95_increase": (
            float(generated_boundary_p95 - input_boundary_p95)
            if input_boundary_p95 is not None else None
        ),
        "support_voxels": int(np.count_nonzero(support_dhw)),
        "outside_voxels": int(np.count_nonzero(outside)),
        "lesion_outside_support_voxels": int(np.count_nonzero(lesion & ~support_dhw)),
        "per_label_histogram": {},
    }
    bin_edges = np.linspace(-1.0, 1.0, 17)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for label_value in (1, 2, 3, 4):
        mask = seg_dhw == label_value
        if not mask.any():
            continue
        generated_hist, _ = np.histogram(generated_dhw[mask], bins=bin_edges)
        generated_hist = generated_hist.astype(np.float64)
        generated_hist /= max(float(generated_hist.sum()), 1.0)
        target = condition[(label_value - 1) * 16 : label_value * 16].astype(np.float64)
        target /= max(float(target.sum()), 1e-12)
        metrics["per_label_histogram"][str(label_value)] = {
            "l1": float(np.abs(generated_hist - target).sum()),
            "wasserstein": float(
                wasserstein_distance(bin_centers, bin_centers, generated_hist, target)
            ),
        }
    return metrics


def _save_qa(
    path: Path,
    original_input_dhw: np.ndarray,
    masked_input_dhw: np.ndarray,
    generated_dhw: np.ndarray,
    seg_dhw: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lesion = seg_dhw > 0
    z = (
        int(np.argmax(lesion.sum(axis=(1, 2))))
        if lesion.any()
        else original_input_dhw.shape[0] // 2
    )
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.6))
    panels = (
        (original_input_dhw[z], "original input", "gray"),
        (masked_input_dhw[z], "masked input", "gray"),
        (generated_dhw[z], "generated output", "gray"),
        (
            np.abs(generated_dhw[z] - original_input_dhw[z]),
            "|output - original|",
            "magma",
        ),
        (lesion[z].astype(np.uint8), "conditioning mask", "gray"),
    )
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_gli(conf: DictConfig) -> None:
    device = dev(conf.get('device'))
    if device.type != 'cuda':
        raise RuntimeError("GLI inference requires CUDA")
    th.cuda.set_device(device)
    th.cuda.reset_peak_memory_stats(device)
    diffusion = build_model_and_diffusion(conf, device, use_data_parallel=False)
    checkpoint = load_diffusion_checkpoint(
        diffusion,
        conf.checkpoint.path,
        weights_key=conf.checkpoint.weights_key,
        expected_metadata=_gli_expected_metadata(conf),
    )
    checkpoint_load_peak_mib = th.cuda.max_memory_allocated(device) / (1024 ** 2)
    checkpoint_sha256 = sha256_file(conf.checkpoint.path)
    checkpoint_git_sha = _checkpoint_git_sha(checkpoint)
    checkpoint_step = int(checkpoint.get("step", -1))
    diffusion.eval()
    condition_source = str(conf.conditioning.source)
    condition_selection = str(conf.conditioning.selection)
    if condition_source not in {"real", "cluster"}:
        raise ValueError(f"unsupported conditioning source: {condition_source}")
    if condition_selection not in {"nearest", "first", "last"}:
        raise ValueError(f"unsupported conditioning selection: {condition_selection}")
    if condition_source == "real" and condition_selection != "nearest":
        raise ValueError("real histogram conditioning uses selection=nearest as a no-op")
    if float(conf.conditioning.get("hist_perturb_std", 0.0)) != 0.0:
        raise ValueError("histogram perturbation must be disabled for formal closed-loop inference")
    if not bool(conf.input_policy.get("mask_inside_lesion", False)):
        raise ValueError("exp007 QA requires input_policy.mask_inside_lesion=true")
    lesion_mode = str(conf.input_policy.lesion_mode)
    if lesion_mode not in {
        "original_multilabel",
        "anchor_label_union",
        "union_single_label_cycle",
    }:
        raise ValueError(f"unsupported exp007 lesion mode: {lesion_mode}")
    lesion_fill_value = float(conf.input_policy.get("fill_value", 0.0))
    centers = load_cluster_centers(conf.conditioning.clusters_path, conf.dataset.patch_size_xyz)
    selection_path = Path(str(conf.selection.manifest_path)).expanduser()
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if str(selection_payload.get("split")) != str(conf.dataset.split):
        raise ValueError("selection manifest split does not match inference config")
    if tuple(selection_payload.get("patch_size_xyz", ())) != tuple(
        int(value) for value in conf.dataset.patch_size_xyz
    ):
        raise ValueError("selection manifest patch shape does not match inference config")
    selected_relative_paths = manifest_shard_paths(
        selection_payload,
        shard_index=int(conf.selection.shard_index),
        shard_count=int(conf.selection.shard_count),
    )
    if int(conf.dataset.batch_size) != 1:
        raise ValueError("formal GLI selected inference requires dataset.batch_size=1")
    loader = get_inference_dataloader(
        dataset_root_dir=conf.dataset.root_dir,
        data_type="gli",
        batch_size=int(conf.dataset.batch_size),
        num_workers=int(conf.dataset.num_workers),
        raw_root_dir=conf.dataset.raw_root_dir,
        patch_size_xyz=conf.dataset.patch_size_xyz,
        split=conf.dataset.split,
        split_file=conf.dataset.split_file,
        raw_source_split=conf.dataset.get('raw_source_split', 'train'),
        selected_relative_paths=selected_relative_paths,
    )
    provenance = selection_payload.get("provenance", {})
    dataset_object = loader.dataset
    if provenance.get("dataset_manifest_sha256") != sha256_file(dataset_object.manifest_path):
        raise ValueError("selection manifest dataset hash mismatch")
    if provenance.get("split_sha256") != sha256_file(conf.dataset.split_file):
        raise ValueError("selection manifest split hash mismatch")
    output_root = _safe_output_root(
        conf.output.root,
        bool(conf.output.get('overwrite', False)),
        bool(conf.output.get('resume', False)),
    )
    run_contract = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "checkpoint_weights_key": str(conf.checkpoint.weights_key),
        "cluster_sha256": sha256_file(conf.conditioning.clusters_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_shard_index": int(conf.selection.shard_index),
        "selection_shard_count": int(conf.selection.shard_count),
        "sampling_seed": int(conf.sampling.seed),
        "conditioning_source": condition_source,
        "conditioning_selection": condition_selection,
        "cond_scale": float(conf.sampling.get("cond_scale", 1.0)),
        "repaint_schedule": OmegaConf.to_container(
            conf.repaint.schedule_jump_params, resolve=True
        ),
        "lesion_channel_label_values": [1, 2, 3, 4],
        "lesion_channel_names": ["NETC", "SNFH", "ET", "RC"],
        "input_policy": {
            "mask_inside_lesion": True,
            "fill_value": lesion_fill_value,
            "lesion_mode": lesion_mode,
            "single_target_label_source": (
                "anchor_label"
                if lesion_mode == "anchor_label_union"
                else ("selection_order_cycle_1_to_4" if lesion_mode == "union_single_label_cycle" else None)
            ),
        },
    }
    contract_path = output_root / "run_contract.json"
    if contract_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise ValueError("existing inference output has a different frozen run contract")
    else:
        temporary_contract = output_root / ".run_contract.json.tmp"
        temporary_contract.write_text(
            json.dumps(run_contract, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_contract, contract_path)
    progress_path = output_root / "progress.jsonl"
    progress_records = _load_progress(progress_path) if bool(conf.output.get('resume', False)) else []
    completed_paths = {
        str(record["manifest"]["source_relative_path"]) for record in progress_records
    }
    unexpected_completed = completed_paths.difference(selected_relative_paths)
    if unexpected_completed:
        raise ValueError(f"progress contains paths outside frozen shard: {sorted(unexpected_completed)[:3]}")
    max_batches_value = conf.output.get('max_batches')
    max_batches = None if max_batches_value is None else int(max_batches_value)
    th.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    memory_trace = []
    new_samples = 0
    sampling_seed = int(conf.sampling.seed)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        relative_path = str(batch['relative_path'][0])
        if relative_path in completed_paths:
            continue
        sample_seed = _sampling_seed(sampling_seed, relative_path)
        th.manual_seed(sample_seed)
        th.cuda.manual_seed_all(sample_seed)
        for key, value in list(batch.items()):
            if isinstance(value, th.Tensor):
                batch[key] = value.to(device)
        original_input = batch['input_t1c']
        source_seg = batch['conditioning_seg']
        source_lesion_mask = batch['lesion_mask']
        masked_input = mask_input_inside_lesion(
            original_input,
            source_seg,
            fill_value=lesion_fill_value,
        )
        target_labels = batch['anchor_label'].to(device=device, dtype=th.long)
        if lesion_mode == "original_multilabel":
            conditioning_seg = source_seg
            lesion_mask = source_lesion_mask
            if condition_source == "real":
                condition = batch['hist'].float()
                cluster_ids = th.full(
                    (condition.shape[0], 4), -1, dtype=th.int64, device=device
                )
            elif condition_selection == "nearest":
                condition, cluster_ids = nearest_cluster_condition(
                    batch['hist'], conditioning_seg, centers
                )
            else:
                condition, cluster_ids = indexed_cluster_condition(
                    batch['hist'],
                    conditioning_seg,
                    centers,
                    index_mode=condition_selection,
                )
        elif lesion_mode == "anchor_label_union":
            conditioning_seg, lesion_mask, condition, cluster_ids = (
                anchor_union_cluster_condition(
                    batch['hist'], source_seg, batch['anchor_label'], centers
                )
            )
        else:
            if condition_source != "cluster" or condition_selection not in {"first", "last"}:
                raise ValueError(
                    "union_single_label_cycle requires cluster first/last conditioning"
                )
            target_labels = th.full_like(
                batch['anchor_label'].to(device=device, dtype=th.long),
                (batch_index % 4) + 1,
            )
            conditioning_seg, lesion_mask, condition, cluster_ids = (
                union_label_cluster_condition(
                    batch['hist'],
                    source_seg,
                    target_labels,
                    centers,
                    index_mode=condition_selection,
                )
            )
        model_gt = masked_input.repeat(1, int(conf.model.diffusion_num_channels), 1, 1, 1)
        model_kwargs = {
            "gt": model_gt,
            "gt_background": masked_input,
            "gt_keep_mask": conditioning_seg,
            "lesion_mask": lesion_mask,
        }
        amp_dtype_name = str(conf.model.get('amp_dtype', 'float16')).lower()
        if amp_dtype_name not in {'float16', 'bfloat16'}:
            raise ValueError(f"unsupported inference AMP dtype: {amp_dtype_name}")
        amp_dtype = th.bfloat16 if amp_dtype_name == 'bfloat16' else th.float16
        with th.autocast(
            device_type="cuda",
            enabled=bool(conf.model.get('amp', False)),
            dtype=amp_dtype,
        ):
            details = diffusion.p_sample_loop_repaint(
                shape=diffusion.sample_shape(batch['GT'].shape[0]),
                model_kwargs=model_kwargs,
                device=device,
                progress=bool(conf.repaint.show_progress),
                conf=conf.repaint,
                cond=condition,
                cond_scale=float(conf.sampling.get("cond_scale", 1.0)),
                return_details=True,
            )
        output = details['sample'].float().cpu()
        channels = details['channels'].float().cpu()
        th.cuda.synchronize(device)
        memory_trace.append(
            {
                "source_relative_path": relative_path,
                "allocated_mib": th.cuda.memory_allocated(device) / (1024 ** 2),
                "reserved_mib": th.cuda.memory_reserved(device) / (1024 ** 2),
                "peak_allocated_mib": th.cuda.max_memory_allocated(device) / (1024 ** 2),
            }
        )
        for index in range(output.shape[0]):
            stem = str(batch['GT_name'][index])
            generated_dhw = output[index, 0].numpy()
            input_dhw = original_input[index, 0].float().cpu().numpy()
            masked_input_dhw = masked_input[index, 0].float().cpu().numpy()
            source_seg_dhw = source_seg[index, 0].cpu().numpy().astype(np.uint8)
            seg_dhw = conditioning_seg[index, 0].cpu().numpy().astype(np.uint8)
            source_lesion_mask_cdhw = (
                source_lesion_mask[index].cpu().numpy().astype(np.uint8)
            )
            lesion_mask_cdhw = lesion_mask[index].cpu().numpy().astype(np.uint8)
            support_dhw = batch['explicit_brain_support_mask'][index, 0].cpu().numpy().astype(bool)
            affine = batch['affine'][index].float().cpu().numpy()
            condition_np = condition[index].float().cpu().numpy()
            cluster_np = cluster_ids[index].cpu().numpy()
            generated_xyz = np.asarray(dhw_to_xyz(generated_dhw), dtype=np.float32)
            input_xyz = np.asarray(dhw_to_xyz(input_dhw), dtype=np.float32)
            masked_input_xyz = np.asarray(dhw_to_xyz(masked_input_dhw), dtype=np.float32)
            source_seg_xyz = np.asarray(dhw_to_xyz(source_seg_dhw), dtype=np.uint8)
            seg_xyz = np.asarray(dhw_to_xyz(seg_dhw), dtype=np.uint8)
            support_xyz = np.asarray(dhw_to_xyz(support_dhw), dtype=np.uint8)
            npz_path = output_root / "generated_npz" / f"{stem}.npz"
            nifti_path = output_root / "generated_nifti" / f"{stem}-generated-t1c.nii.gz"
            seg_path = output_root / "generated_nifti" / f"{stem}-conditioning-seg.nii.gz"
            np.savez_compressed(
                npz_path,
                generated_t1c_xyz=generated_xyz,
                input_t1c_xyz=input_xyz,
                original_input_t1c_xyz=input_xyz,
                masked_input_t1c_xyz=masked_input_xyz,
                source_conditioning_seg_xyz=source_seg_xyz,
                conditioning_seg_xyz=seg_xyz,
                explicit_brain_support_mask_xyz=support_xyz,
                generated_channels_cdhw=channels[index].numpy().astype(np.float32),
                source_lesion_mask_cdhw=source_lesion_mask_cdhw,
                lesion_mask_cdhw=lesion_mask_cdhw,
                condition_hist_64=condition_np,
                cluster_ids=cluster_np,
                affine=affine,
                checkpoint_git_commit=np.asarray(
                    checkpoint_git_sha
                ),
                checkpoint_step=np.asarray(checkpoint_step, dtype=np.int64),
                checkpoint_sha256=np.asarray(checkpoint_sha256),
                checkpoint_weights_key=np.asarray(str(conf.checkpoint.weights_key)),
                lesion_channel_label_values=np.asarray([1, 2, 3, 4], dtype=np.uint8),
                lesion_channel_names=np.asarray(["NETC", "SNFH", "ET", "RC"]),
                sample_seed=np.asarray(sample_seed, dtype=np.int64),
                input_lesion_mode=np.asarray(lesion_mode),
                lesion_fill_value=np.asarray(lesion_fill_value, dtype=np.float32),
                target_anchor_label=np.asarray(
                    int(target_labels[index].item()), dtype=np.uint8
                ),
            )
            nib.save(nib.Nifti1Image(generated_xyz, affine), nifti_path)
            nib.save(nib.Nifti1Image(seg_xyz, affine), seg_path)
            reloaded = nib.load(str(nifti_path))
            expected_xyz = tuple(int(value) for value in conf.dataset.patch_size_xyz)
            if reloaded.shape != expected_xyz or not np.allclose(reloaded.affine, affine):
                raise RuntimeError(f"NIfTI round-trip failed for {stem}")
            metrics = _region_metrics(
                generated_dhw, input_dhw, seg_dhw, support_dhw, condition_np
            )
            metrics.update(
                {
                    "case_id": str(batch['case_id'][index]),
                    "source_relative_path": str(batch['relative_path'][index]),
                    "cluster_ids": cluster_np.tolist(),
                    "output_shape_dhw": list(generated_dhw.shape),
                    "output_shape_xyz": list(generated_xyz.shape),
                    "internal_channels_shape_cdhw": list(channels[index].shape),
                    "lesion_channel_voxels": lesion_mask_cdhw.reshape(4, -1).sum(axis=1).astype(int).tolist(),
                    "source_lesion_channel_voxels": source_lesion_mask_cdhw.reshape(4, -1).sum(axis=1).astype(int).tolist(),
                    "input_lesion_mode": lesion_mode,
                    "target_anchor_label": int(target_labels[index].item()),
                    "masked_input_lesion_max_abs": float(
                        np.abs(masked_input_dhw[seg_dhw > 0]).max()
                    ),
                    "model_calls": int(details['model_calls']),
                    "sample_seed": sample_seed,
                }
            )
            _save_qa(
                output_root / "qa" / f"{stem}.png",
                input_dhw,
                masked_input_dhw,
                generated_dhw,
                seg_dhw,
            )
            manifest_record = {
                "case_id": str(batch['case_id'][index]),
                "source_relative_path": str(batch['relative_path'][index]),
                "generated_npz": str(npz_path.relative_to(output_root)),
                "generated_nifti": str(nifti_path.relative_to(output_root)),
                "conditioning_seg_nifti": str(seg_path.relative_to(output_root)),
                "cluster_ids": ",".join(map(str, cluster_np.tolist())),
                "checkpoint_git_sha": checkpoint_git_sha,
                "checkpoint_step": checkpoint_step,
                "checkpoint_weights_key": str(conf.checkpoint.weights_key),
                "input_lesion_mode": lesion_mode,
                "target_anchor_label": int(target_labels[index].item()),
            }
            progress_record = {"manifest": manifest_record, "metrics": metrics}
            _append_progress(progress_path, progress_record)
            progress_records.append(progress_record)
            completed_paths.add(relative_path)
            new_samples += 1
    manifests = [record["manifest"] for record in progress_records]
    all_metrics = [record["metrics"] for record in progress_records]
    if not manifests:
        raise RuntimeError("GLI inference produced no samples")
    if max_batches is None and completed_paths != set(selected_relative_paths):
        missing = sorted(set(selected_relative_paths).difference(completed_paths))
        raise RuntimeError(f"GLI inference did not complete frozen shard: {missing[:3]}")
    elapsed = time.perf_counter() - started
    peak_memory_mib = th.cuda.max_memory_allocated(device) / (1024 ** 2)
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests[0]))
        writer.writeheader()
        writer.writerows(manifests)
    tolerance_mib = float(conf.output.get("memory_growth_tolerance_mib", 256.0))
    allocated_values = [float(item["allocated_mib"]) for item in memory_trace]
    reserved_values = [float(item["reserved_mib"]) for item in memory_trace]
    allocated_growth = (
        max(allocated_values[-3:]) - min(allocated_values[-3:]) if len(allocated_values) >= 3 else 0.0
    )
    reserved_growth = (
        max(reserved_values[-3:]) - min(reserved_values[-3:]) if len(reserved_values) >= 3 else 0.0
    )
    memory_stable = allocated_growth <= tolerance_mib and reserved_growth <= tolerance_mib
    total_model_calls = sum(int(metrics["model_calls"]) for metrics in all_metrics)
    summary = {
        "experiment_id": conf.experiment_id,
        "variant": conf.variant,
        "sample_count": len(manifests),
        "elapsed_seconds": elapsed,
        "new_samples": new_samples,
        "peak_memory_mib": peak_memory_mib,
        "checkpoint_load_peak_memory_mib": checkpoint_load_peak_mib,
        "memory_trace": memory_trace,
        "memory_growth_tolerance_mib": tolerance_mib,
        "last_three_allocated_span_mib": allocated_growth,
        "last_three_reserved_span_mib": reserved_growth,
        "memory_stable": memory_stable,
        "model_calls": total_model_calls,
        "model_calls_per_sample": int(all_metrics[0]["model_calls"]),
        "checkpoint_schema_version": int(checkpoint.get("schema_version", -1)),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_git_sha": checkpoint_git_sha,
        "checkpoint_weights_key": str(conf.checkpoint.weights_key),
        "cluster_sha256": sha256_file(conf.conditioning.clusters_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_shard_index": int(conf.selection.shard_index),
        "selection_shard_count": int(conf.selection.shard_count),
        "lesion_channel_label_values": [1, 2, 3, 4],
        "lesion_channel_names": ["NETC", "SNFH", "ET", "RC"],
        "input_policy": run_contract["input_policy"],
        "samples": all_metrics,
    }
    temporary_metrics = output_root / ".metrics.json.tmp"
    temporary_metrics.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_metrics, output_root / "metrics.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


@hydra.main(config_path='confs', config_name='infer', version_base=None)
def main(conf: DictConfig):
    data_type = conf['data_type'].lower()
    if data_type == 'gli':
        run_gli(conf)
        return
    if data_type not in ['lidc', 'emidec']:
        raise ValueError("Wrong data type")
    print("Start", data_type)
    device = dev(conf.get('device'))
    spatial_shape = normalize_spatial_shape(
        conf.get('spatial_shape_dhw'),
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
    )
    base_dim = conf.get('base_dim') or conf.diffusion_img_size

    model = Unet3D(
        dim=base_dim,
        dim_mults=conf.dim_mults,
        channels=conf.diffusion_num_channels,
        cond_dim=conf.cond_dim,
        temporal_max_distance=conf.get('temporal_max_distance', 32),
    )

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        spatial_shape=spatial_shape,
        channels=conf.diffusion_num_channels,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
        data_type=data_type,
    )
    diffusion.to(device)

    weights_dict = {}
    for k, v in (load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")["model"].items()):
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    diffusion.load_state_dict(weights_dict)

    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'hist_clusters', f'{data_type}_clusters.json')
    with open(file_path, 'r') as f:
        clusters = json.load(f)

    cluster_centers = clusters[0]['centers']

    print("sampling...")

    dl = get_inference_dataloader(dataset_root_dir=conf.dataset_root_dir, test_txt_dir=conf.test_txt_dir, batch_size=conf.batch_size, data_type=data_type)  
    
    idx = 0
    for batch in iter(dl):
        for type in range(conf.types):
            print("idx:",idx+1)
            print("type_of_cond:", type+1)
            if data_type == 'lidc':
                hist = th.tensor(cluster_centers[type])
                hist = perturb_tensor(tensor=hist)
                hist = hist.unsqueeze(0)
            elif data_type == 'emidec':
                hist_1 = perturb_tensor(th.tensor(cluster_centers[0]))
                hist_2 = perturb_tensor(th.tensor(cluster_centers[1]))
                hist = th.cat((hist_1, hist_2), dim=0).to(device)
            for k in batch.keys():
                if isinstance(batch[k], th.Tensor):
                    batch[k] = batch[k].to(device)
            model_kwargs = {}
            model_kwargs["gt"] = batch['GT']
            gt_keep_mask = batch.get('gt_keep_mask')
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask
            batch_size = model_kwargs["gt"].shape[0]

            sample_fn = diffusion.p_sample_loop_repaint

            output = sample_fn(
                shape=diffusion.sample_shape(batch_size),
                model_kwargs=model_kwargs,
                device=device,
                progress=show_progress,
                conf=conf,
                cond=hist
            )

            if data_type == 'lidc':
                image_fold = f"Image_{type+1}"
                label_fold = f"Mask_{type+1}"
                os.makedirs(os.path.join(conf.target_img_path, image_fold), exist_ok=True)
                os.makedirs(os.path.join(conf.target_label_path, label_fold), exist_ok=True)
                for i in range(batch_size):
                    result = output[i, :, :, :, :].cpu()
                    restore_affine = batch['affine'][i].squeeze(0).cpu()
                    gt_name = batch['GT_name'][i]
                    name_part, extension = gt_name.rsplit('.nii.gz', 1)[0], '.nii.gz'
                    main_name, vol_part = name_part.rsplit('_CVol_', 1)
                    mask_name = f"{main_name}_Mask_{vol_part}{extension}"
                    gen_image = tio.ScalarImage(tensor=result, channels_last=False, affine=restore_affine)
                    gen_image.save(os.path.join(conf.target_img_path, image_fold, gt_name))
                    label = batch['gt_keep_mask'][i].cpu()
                    label = tio.LabelMap(tensor=label, channels_last=False, affine=restore_affine)
                    label.save(os.path.join(conf.target_label_path, label_fold, mask_name))
            elif data_type == 'emidec':
                output = output.permute(0, 1, 3, 4, 2).cpu()
                os.makedirs(conf.target_img_path, exist_ok=True)
                os.makedirs(conf.target_label_path, exist_ok=True)
                for i in range(batch_size):
                    result = output[i, :, :, :, :].cpu()
                    restore_affine = batch['affine'][i].squeeze(0).cpu()
                    gt_name = batch['GT_name'][i]
                    gen_image = tio.ScalarImage(tensor=result, channels_last=False, affine=restore_affine)
                    gen_image.save(os.path.join(conf.target_img_path, gt_name))
                    label = batch['gt_keep_mask'][i].cpu()
                    label = tio.LabelMap(tensor=label, channels_last=False, affine=restore_affine)
                    label.save(os.path.join(conf.target_label_path, gt_name))


        idx += 1

    print("sampling complete")


if __name__ == "__main__":
    main()

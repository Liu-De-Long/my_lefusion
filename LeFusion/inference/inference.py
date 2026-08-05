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
from checkpointing import load_diffusion_checkpoint
from inference.gli_utils import dhw_to_xyz, load_cluster_centers, nearest_cluster_condition
import torchio as tio
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance
import yaml
from omegaconf import DictConfig
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
    }


def _safe_output_root(path: str, overwrite: bool) -> Path:
    output = Path(path).expanduser()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"GLI inference output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("generated_npz", "generated_nifti", "qa"):
        (output / name).mkdir(exist_ok=True)
    return output


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
    lesion_values = difference[lesion]
    shell = ndimage.binary_dilation(lesion, iterations=1) & ~lesion & support_dhw
    metrics = {
        "healthy_brain_mae": float(healthy_values.mean()) if healthy_values.size else None,
        "healthy_brain_max_abs": float(healthy_values.max()) if healthy_values.size else None,
        "healthy_brain_exact": bool(not healthy_values.size or np.all(healthy_values == 0)),
        "outside_mean_abs": float(outside_values.mean()) if outside_values.size else None,
        "outside_max_abs": float(outside_values.max()) if outside_values.size else None,
        "lesion_change_mae": float(lesion_values.mean()) if lesion_values.size else None,
        "boundary_outer_shell_mae": float(difference[shell].mean()) if shell.any() else None,
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
    input_dhw: np.ndarray,
    generated_dhw: np.ndarray,
    seg_dhw: np.ndarray,
    support_dhw: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lesion = seg_dhw > 0
    z = int(np.argmax(lesion.sum(axis=(1, 2)))) if lesion.any() else input_dhw.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    panels = (
        (input_dhw[z], "input"),
        (generated_dhw[z], "generated"),
        (np.abs(generated_dhw[z] - input_dhw[z]), "absolute difference"),
        (input_dhw[z], "seg/support overlay"),
    )
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, cmap="magma" if "difference" in title else "gray", origin="lower")
        axis.set_title(title)
        axis.axis("off")
    if lesion[z].any():
        axes[3].contour(lesion[z], levels=[0.5], colors="red", linewidths=0.8, origin="lower")
    if support_dhw[z].any():
        axes[3].contour(support_dhw[z], levels=[0.5], colors="cyan", linewidths=0.5, origin="lower")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_gli(conf: DictConfig) -> None:
    device = dev(conf.get('device'))
    if device.type != 'cuda':
        raise RuntimeError("GLI inference smoke requires CUDA")
    th.cuda.set_device(device)
    diffusion = build_model_and_diffusion(conf, device, use_data_parallel=False)
    checkpoint = load_diffusion_checkpoint(
        diffusion,
        conf.checkpoint.path,
        weights_key=conf.checkpoint.weights_key,
        expected_metadata=_gli_expected_metadata(conf),
    )
    diffusion.eval()
    if str(conf.conditioning.source) != "cluster":
        raise ValueError("GLI closed-loop smoke requires conditioning.source=cluster")
    if str(conf.conditioning.selection) != "nearest":
        raise ValueError("GLI closed-loop smoke requires conditioning.selection=nearest")
    if float(conf.conditioning.get("hist_perturb_std", 0.0)) != 0.0:
        raise ValueError("histogram perturbation must be disabled for the first closed-loop smoke")
    centers = load_cluster_centers(conf.conditioning.clusters_path, conf.dataset.patch_size_xyz)
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
    )
    output_root = _safe_output_root(conf.output.root, bool(conf.output.get('overwrite', False)))
    manifests = []
    all_metrics = []
    max_batches = int(conf.output.max_batches)
    th.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_model_calls = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        for key, value in list(batch.items()):
            if isinstance(value, th.Tensor):
                batch[key] = value.to(device)
        condition, cluster_ids = nearest_cluster_condition(
            batch['hist'], batch['conditioning_seg'], centers
        )
        model_kwargs = {
            "gt": batch['GT'],
            "gt_background": batch['input_t1c'],
            "gt_keep_mask": batch['conditioning_seg'],
            "lesion_mask": batch['lesion_mask'],
        }
        with th.autocast(device_type="cuda", enabled=bool(conf.model.get('amp', False))):
            details = diffusion.p_sample_loop_repaint(
                shape=diffusion.sample_shape(batch['GT'].shape[0]),
                model_kwargs=model_kwargs,
                device=device,
                progress=bool(conf.repaint.show_progress),
                conf=conf.repaint,
                cond=condition,
                return_details=True,
            )
        output = details['sample'].float().cpu()
        channels = details['channels'].float().cpu()
        total_model_calls += int(details['model_calls'])
        for index in range(output.shape[0]):
            stem = str(batch['GT_name'][index])
            generated_dhw = output[index, 0].numpy()
            input_dhw = batch['input_t1c'][index, 0].float().cpu().numpy()
            seg_dhw = batch['conditioning_seg'][index, 0].cpu().numpy().astype(np.uint8)
            lesion_mask_cdhw = batch['lesion_mask'][index].cpu().numpy().astype(np.uint8)
            support_dhw = batch['explicit_brain_support_mask'][index, 0].cpu().numpy().astype(bool)
            affine = batch['affine'][index].float().cpu().numpy()
            condition_np = condition[index].float().cpu().numpy()
            cluster_np = cluster_ids[index].cpu().numpy()
            generated_xyz = np.asarray(dhw_to_xyz(generated_dhw), dtype=np.float32)
            input_xyz = np.asarray(dhw_to_xyz(input_dhw), dtype=np.float32)
            seg_xyz = np.asarray(dhw_to_xyz(seg_dhw), dtype=np.uint8)
            support_xyz = np.asarray(dhw_to_xyz(support_dhw), dtype=np.uint8)
            npz_path = output_root / "generated_npz" / f"{stem}.npz"
            nifti_path = output_root / "generated_nifti" / f"{stem}-generated-t1c.nii.gz"
            seg_path = output_root / "generated_nifti" / f"{stem}-conditioning-seg.nii.gz"
            np.savez_compressed(
                npz_path,
                generated_t1c_xyz=generated_xyz,
                input_t1c_xyz=input_xyz,
                conditioning_seg_xyz=seg_xyz,
                explicit_brain_support_mask_xyz=support_xyz,
                generated_channels_cdhw=channels[index].numpy().astype(np.float32),
                lesion_mask_cdhw=lesion_mask_cdhw,
                condition_hist_64=condition_np,
                cluster_ids=cluster_np,
                affine=affine,
                checkpoint_git_commit=np.asarray(
                    str(checkpoint["metadata"].get("git_commit", "unknown"))
                ),
                checkpoint_weights_key=np.asarray(str(conf.checkpoint.weights_key)),
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
                }
            )
            all_metrics.append(metrics)
            _save_qa(
                output_root / "qa" / f"{stem}.png",
                input_dhw,
                generated_dhw,
                seg_dhw,
                support_dhw,
            )
            manifests.append(
                {
                    "case_id": str(batch['case_id'][index]),
                    "source_relative_path": str(batch['relative_path'][index]),
                    "generated_npz": str(npz_path.relative_to(output_root)),
                    "generated_nifti": str(nifti_path.relative_to(output_root)),
                    "conditioning_seg_nifti": str(seg_path.relative_to(output_root)),
                    "cluster_ids": ",".join(map(str, cluster_np.tolist())),
                    "checkpoint_git_commit": str(
                        checkpoint["metadata"].get("git_commit", "unknown")
                    ),
                    "checkpoint_weights_key": str(conf.checkpoint.weights_key),
                }
            )
    if not manifests:
        raise RuntimeError("GLI inference produced no samples")
    elapsed = time.perf_counter() - started
    peak_memory_mib = th.cuda.max_memory_allocated(device) / (1024 ** 2)
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests[0]))
        writer.writeheader()
        writer.writerows(manifests)
    summary = {
        "experiment_id": conf.experiment_id,
        "variant": conf.variant,
        "sample_count": len(manifests),
        "elapsed_seconds": elapsed,
        "peak_memory_mib": peak_memory_mib,
        "model_calls": total_model_calls,
        "checkpoint_git_commit": str(
            checkpoint["metadata"].get("git_commit", "unknown")
        ),
        "checkpoint_weights_key": str(conf.checkpoint.weights_key),
        "samples": all_metrics,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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

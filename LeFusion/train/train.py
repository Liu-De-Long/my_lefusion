import sys
import os
import random
import subprocess
from pathlib import Path

import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, Trainer, GaussianDiffusion_Nolatent
import hydra
from omegaconf import DictConfig, OmegaConf
from get_dataset.get_dataset import get_train_dataset, get_validation_dataset
import torch
from ddpm.unet import UNet
import torch.nn as nn
from checkpointing import canonical_config_hash, sha256_file
from train.tracking import initialize_wandb, validate_wandb_config


SUPPORTED_DATA_TYPES = ('lidc', 'emidec', 'gli')


def resolve_spatial_shape(cfg: DictConfig) -> tuple[int, int, int]:
    explicit_shape = cfg.model.get('spatial_shape_dhw')
    if explicit_shape is not None:
        shape = tuple(int(value) for value in explicit_shape)
    else:
        shape = (
            int(cfg.model.diffusion_depth_size),
            int(cfg.model.diffusion_img_size),
            int(cfg.model.diffusion_img_size),
        )
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"model.spatial_shape_dhw must be three positive integers: {shape}")
    return shape


def validate_training_config(cfg: DictConfig) -> tuple[str, tuple[int, int, int]]:
    data_type = cfg.dataset.data_type.lower()
    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(f"Wrong data type: {data_type}")
    spatial_shape = resolve_spatial_shape(cfg)
    downsample_factor = 2 ** (len(cfg.model.dim_mults) - 1)
    if spatial_shape[1] % downsample_factor or spatial_shape[2] % downsample_factor:
        raise ValueError(
            f"height and width must be divisible by {downsample_factor}: {spatial_shape}"
        )
    if data_type == 'gli':
        if int(cfg.model.diffusion_num_channels) != 4:
            raise ValueError("GLI training requires model.diffusion_num_channels=4")
        if int(cfg.model.cond_dim) != 64:
            raise ValueError("GLI training requires model.cond_dim=64")
        spatial_condition_channels = int(
            cfg.model.get('spatial_condition_channels', 0)
        )
        if spatial_condition_channels not in {0, 5}:
            raise ValueError(
                "GLI spatial conditioning must be disabled (0) or use "
                "masked T1c + four masks (5 channels)"
            )
        inpainting_cfg = cfg.get('inpainting_training')
        if inpainting_cfg is not None and bool(inpainting_cfg.get('enabled', False)):
            if spatial_condition_channels != 5:
                raise ValueError(
                    "conditional inpainting requires model.spatial_condition_channels=5"
                )
            if float(inpainting_cfg.get('fill_value', 0.0)) != 0.0:
                raise ValueError("the frozen exp008 lesion fill value must be 0.0")
        patch_xyz = tuple(int(value) for value in cfg.dataset.patch_size_xyz)
        expected_dhw = (patch_xyz[2], patch_xyz[0], patch_xyz[1])
        if spatial_shape != expected_dhw:
            raise ValueError(
                "GLI patch/model shape mismatch: "
                f"patch_size_xyz={patch_xyz} maps to {expected_dhw}, got {spatial_shape}"
            )
        formal_cfg = cfg.get('formal_training')
        if formal_cfg is not None and bool(formal_cfg.get('enabled', False)):
            effective_batch = int(cfg.model.batch_size) * int(
                cfg.model.gradient_accumulate_every
            )
            if effective_batch != int(formal_cfg.effective_batch_size):
                raise ValueError(
                    f"effective batch mismatch: {effective_batch} != "
                    f"{formal_cfg.effective_batch_size}"
                )
            if int(cfg.model.train_num_steps) <= 0:
                raise ValueError("formal max optimizer steps must be positive")
            if float(cfg.model.max_grad_norm) <= 0:
                raise ValueError("formal gradient clipping must be positive")
            if not bool(cfg.validation.enabled) or str(cfg.validation.split) != 'val':
                raise ValueError("formal training requires supervised split='val' validation")
            if str(cfg.sampler.name) != 'gli_anchor_role_subject':
                raise ValueError("formal GLI training requires the approved stratified sampler")
            resume_from = cfg.checkpoint.get('resume_from')
            expected_resume = 'must' if resume_from else 'never'
            if str(cfg.wandb.resume) != expected_resume:
                raise ValueError(
                    f"wandb.resume must be {expected_resume!r} for this checkpoint mode"
                )
            validate_wandb_config(cfg)
    return data_type, spatial_shape


def build_model_and_diffusion(
    cfg: DictConfig,
    device: torch.device,
    *,
    use_data_parallel: bool = True,
):
    data_type, spatial_shape = validate_training_config(cfg)
    base_dim = cfg.model.get('base_dim')
    if base_dim is None:
        base_dim = cfg.model.diffusion_img_size

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=int(base_dim),
            dim_mults=tuple(cfg.model.dim_mults),
            channels=int(cfg.model.diffusion_num_channels),
            spatial_condition_channels=int(
                cfg.model.get('spatial_condition_channels', 0)
            ),
            cond_dim=int(cfg.model.cond_dim),
            temporal_max_distance=int(cfg.model.get('temporal_max_distance', 32)),
        )
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3,
        )
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    model = model.to(device)
    if use_data_parallel and device.type == 'cuda' and torch.cuda.device_count() > 1:
        device_ids = cfg.model.get('data_parallel_device_ids')
        model = nn.DataParallel(
            model,
            device_ids=None if device_ids is None else [int(value) for value in device_ids],
        )

    return GaussianDiffusion_Nolatent(
        model,
        image_size=int(cfg.model.diffusion_img_size),
        num_frames=int(cfg.model.diffusion_depth_size),
        spatial_shape=spatial_shape,
        channels=int(cfg.model.diffusion_num_channels),
        timesteps=int(cfg.model.timesteps),
        loss_type=cfg.model.loss_type,
        device=device,
        data_type=data_type,
    ).to(device)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_git_sha() -> str:
    configured = os.environ.get('GIT_COMMIT')
    if configured:
        return configured
    repository = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repository, text=True
    ).strip()


def build_checkpoint_metadata(cfg, train_dataset, spatial_shape):
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    split_file = Path(str(cfg.dataset.split_file)).expanduser()
    run_id = cfg.wandb.get('run_id')
    return resolved, {
        'experiment_id': str(cfg.experiment_id),
        'git_sha': resolve_git_sha(),
        'config_hash': canonical_config_hash(resolved),
        'manifest_hash': sha256_file(train_dataset.manifest_path),
        'split_hash': sha256_file(split_file),
        'data_type': str(cfg.dataset.data_type),
        'patch_size_xyz': [int(value) for value in cfg.dataset.patch_size_xyz],
        'spatial_shape_dhw': list(spatial_shape),
        'sampler_name': str(cfg.sampler.name),
        'wandb_run_id': None if run_id is None else str(run_id),
        'spatial_condition_channels': int(
            cfg.model.get('spatial_condition_channels', 0)
        ),
    }


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    data_type, spatial_shape = validate_training_config(cfg)
    set_global_seed(int(cfg.get('seed', 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.model.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = build_model_and_diffusion(
        cfg,
        device,
        use_data_parallel=bool(cfg.model.get('data_parallel', True)),
    )

    train_dataset, train_sampler = get_train_dataset(cfg)
    validation_dataset = get_validation_dataset(cfg)
    resolved_config = None
    checkpoint_metadata = None
    if data_type == 'gli' and cfg.get('formal_training', {}).get('enabled', False):
        resolved_config, checkpoint_metadata = build_checkpoint_metadata(
            cfg, train_dataset, spatial_shape
        )

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        device=device,
        max_grad_norm=cfg.model.get('max_grad_norm'),
        train_sampler=train_sampler,
        validation_dataset=validation_dataset,
        validation_config=cfg.get('validation'),
        checkpoint_config=cfg.get('checkpoint'),
        checkpoint_metadata=checkpoint_metadata,
        resolved_config=resolved_config,
    )

    resume_from = cfg.get('checkpoint', {}).get('resume_from')
    if resume_from:
        trainer.load(resume_from)
    elif cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    wandb_run = initialize_wandb(cfg)
    try:
        if wandb_run is None:
            trainer.train()
        else:
            trainer.train(log_fn=wandb_run.log)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    run()

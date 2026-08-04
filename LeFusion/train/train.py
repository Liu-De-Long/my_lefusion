import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, Trainer, GaussianDiffusion_Nolatent
import hydra
from omegaconf import DictConfig, OmegaConf
from get_dataset.get_dataset import get_train_dataset
import torch
from ddpm.unet import UNet
import torch.nn as nn


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
        patch_xyz = tuple(int(value) for value in cfg.dataset.patch_size_xyz)
        expected_dhw = (patch_xyz[2], patch_xyz[0], patch_xyz[1])
        if spatial_shape != expected_dhw:
            raise ValueError(
                "GLI patch/model shape mismatch: "
                f"patch_size_xyz={patch_xyz} maps to {expected_dhw}, got {spatial_shape}"
            )
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
    if use_data_parallel and device.type == 'cuda':
        model = nn.DataParallel(model)

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


def initialize_wandb(cfg: DictConfig):
    wandb_cfg = cfg.get('wandb')
    if wandb_cfg is None or not bool(wandb_cfg.get('enabled', False)):
        return None
    import wandb

    entity = wandb_cfg.get('entity') or os.environ.get('WANDB_ENTITY')
    wandb_dir = wandb_cfg.get('dir', cfg.model.results_folder)
    os.makedirs(wandb_dir, exist_ok=True)
    return wandb.init(
        project=wandb_cfg.get('project', 'lefusion-brats2024-gli'),
        entity=entity,
        name=wandb_cfg.get('run_name'),
        mode=wandb_cfg.get('mode', 'online'),
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=wandb_dir,
    )


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.model.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = build_model_and_diffusion(cfg, device)

    train_dataset, *_ = get_train_dataset(cfg)

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
    )

    if cfg.model.load_milestone:
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

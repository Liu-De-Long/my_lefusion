from torch.utils.data import DataLoader
from dataset import LIDCDataset, LIDCInDataset
from dataset import EMIDECDataset, EMIDECInDataset
from dataset import GLIDataset, GLIInferenceDataset


def get_inference_dataloader(
    dataset_root_dir,
    test_txt_dir='',
    batch_size=1,
    drop_last=False,
    data_type='',
    num_workers=2,
    **dataset_kwargs,
):
    if data_type == 'lidc':
        train_dataset = LIDCInDataset(root_dir=dataset_root_dir, test_txt_dir=test_txt_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )
    elif data_type == 'emidec':
        train_dataset = EMIDECInDataset(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )
    elif data_type == 'gli':
        train_dataset = GLIInferenceDataset(
            root_dir=dataset_root_dir,
            raw_root_dir=dataset_kwargs['raw_root_dir'],
            patch_size_xyz=dataset_kwargs['patch_size_xyz'],
            split=dataset_kwargs.get('split', 'test'),
            split_file=dataset_kwargs.get('split_file'),
            raw_source_split=dataset_kwargs.get('raw_source_split', 'train'),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    else:
        raise ValueError(f"Wrong data type: {data_type}")
    return loader

def get_train_dataset(cfg):
    if cfg.dataset.data_type == 'lidc':
        train_dataset = LIDCDataset(root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        sampler = None
    elif cfg.dataset.data_type == 'emidec':
        train_dataset = EMIDECDataset(root_dir=cfg.dataset.root_dir)
        sampler = None
    elif cfg.dataset.data_type == 'gli':
        train_dataset = GLIDataset(
            root_dir=cfg.dataset.root_dir,
            patch_size_xyz=cfg.dataset.patch_size_xyz,
            split=cfg.dataset.get('split', 'train'),
            split_file=cfg.dataset.get('split_file'),
        )
        sampler = None
    else:
        raise ValueError(f"Wrong data type: {cfg.dataset.data_type}")
    return train_dataset, sampler

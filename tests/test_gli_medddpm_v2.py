import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "gli_medddpm_v2.py"
SPEC = importlib.util.spec_from_file_location("gli_medddpm_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_xyz_to_cdhw_preserves_voxel_order():
    xyz = np.arange(64 * 64 * 32, dtype=np.float32).reshape(64, 64, 32)
    cdhw = MODULE.xyz_to_cdhw(xyz)
    assert cdhw.shape == (1, 32, 64, 64)
    assert cdhw[0, 7, 11, 13] == xyz[11, 13, 7]


def test_state_dict_prefix_normalization():
    value = MODULE.torch.ones(1)
    single = MODULE.normalize_state_dict({"denoise_fn.module.x": value}, multi_gpu=False)
    assert set(single) == {"denoise_fn.x"}
    multi = MODULE.normalize_state_dict({"denoise_fn.x": value}, multi_gpu=True)
    assert set(multi) == {"denoise_fn.module.x"}

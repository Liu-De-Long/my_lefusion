from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_population(selector):
    rows = []
    index = 0
    for label, role in selector.STRATA:
        for candidate in range(14):
            rows.append(
                {
                    "anchor_label": label,
                    "sample_role": role,
                    "subject_id": f"P{index:03d}",
                    "relative_path": f"label{label}/{role}/P{index:03d}_{candidate}.npz",
                    "volume_percentile": (index + 0.5) / 112.0,
                }
            )
            index += 1
    return rows


def test_exp022_selection_is_reproducible_patient_distinct_and_exact():
    selector = load_script("gli_exp022_select_test10.py")
    population = synthetic_population(selector)
    first = selector.select_test10(population, seed=20260806, beam_width=1000)
    second = selector.select_test10(population, seed=20260806, beam_width=1000)
    assert [row["relative_path"] for row in first] == [row["relative_path"] for row in second]
    assert len(first) == 10
    assert len({row["subject_id"] for row in first}) == 10
    assert Counter(int(row["anchor_label"]) for row in first) == {1: 2, 2: 4, 3: 2, 4: 2}
    assert Counter(str(row["sample_role"]) for row in first) == {"boundary": 5, "interior": 5}
    assert Counter((int(row["anchor_label"]), str(row["sample_role"])) for row in first) == selector.STRATUM_QUOTAS


def test_exp022_domain_and_axis_adapters():
    evaluator = load_script("gli_exp022_evaluate_and_figure.py")
    cdhw = np.arange(1 * 32 * 64 * 64, dtype=np.float32).reshape(1, 32, 64, 64)
    xyz = evaluator.cdhw_to_xyz(cdhw)
    assert xyz.shape == (64, 64, 32)
    assert np.array_equal(np.transpose(xyz, (2, 0, 1))[None], cdhw)
    real = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    assert evaluator.psnr(real, real) == float("inf")
    assert evaluator.hist_w1(real, real) == 0.0

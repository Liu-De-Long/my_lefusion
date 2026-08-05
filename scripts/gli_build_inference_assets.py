"""Build leakage-safe GLI splits and train-only histogram cluster assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


LABELS = ((1, "netc"), (2, "snfh"), (3, "et"), (4, "rc"))
HIST_BINS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, value: object) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_split_v2(
    source_split_file: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    seed: int,
    val_count: int = 73,
) -> dict:
    source = json.loads(source_split_file.read_text(encoding="utf-8"))
    original = {str(k): str(v) for k, v in source["subject_split"].items()}
    rows = _read_manifest(manifest_path)
    presence: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        presence[row["subject_id"]].add(int(row["anchor_label"]))

    train_subjects = sorted(subject for subject, split in original.items() if split == "train")
    old_val_subjects = sorted(subject for subject, split in original.items() if split == "val")
    if len(train_subjects) != 584 or len(old_val_subjects) != 147:
        raise ValueError(
            "expected the published 584/147 subject split, got "
            f"{len(train_subjects)}/{len(old_val_subjects)}"
        )
    if not 0 < val_count < len(old_val_subjects):
        raise ValueError(f"invalid val_count={val_count}")

    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for subject in old_val_subjects:
        signature = tuple(int(label in presence[subject]) for label, _ in LABELS)
        strata[signature].append(subject)

    allocations: dict[tuple[int, ...], int] = {}
    fractional: list[tuple[float, tuple[int, ...]]] = []
    for signature, subjects in strata.items():
        ideal = len(subjects) * val_count / len(old_val_subjects)
        allocations[signature] = math.floor(ideal)
        fractional.append((ideal - math.floor(ideal), signature))
    remainder = val_count - sum(allocations.values())
    fractional.sort(key=lambda item: (-item[0], item[1]))
    for _, signature in fractional[:remainder]:
        allocations[signature] += 1

    subject_split = {subject: "train" for subject in train_subjects}
    for signature, subjects in sorted(strata.items()):
        ordered = np.asarray(sorted(subjects), dtype=object)
        rng = np.random.default_rng(_stable_seed(seed, signature))
        rng.shuffle(ordered)
        current_val = set(ordered[: allocations[signature]].tolist())
        for subject in subjects:
            subject_split[subject] = "val" if subject in current_val else "test"

    counts = Counter(subject_split.values())
    expected = {"train": 584, "val": 73, "test": 74}
    if dict(counts) != expected:
        raise RuntimeError(f"split count mismatch: {dict(counts)} != {expected}")
    payload = {
        "schema_version": 2,
        "seed": seed,
        "strategy": "preserve_train_stratified_half_of_original_val",
        "source_split_sha256": _sha256(source_split_file),
        "manifest_sha256": _sha256(manifest_path),
        "subject_counts": expected,
        "subject_split": dict(sorted(subject_split.items())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _effective_split(row: dict[str, str], split_map: dict[str, str]) -> str:
    try:
        return split_map[row["subject_id"]]
    except KeyError as exc:
        raise ValueError(f"subject missing from split map: {row['subject_id']}") from exc


def _subject_weights(subject_ids: list[str]) -> np.ndarray:
    counts = Counter(subject_ids)
    return np.asarray([1.0 / counts[subject] for subject in subject_ids], dtype=np.float64)


def _fit_label_clusters(
    x: np.ndarray,
    subject_ids: list[str],
    *,
    seed: int,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    if x.ndim != 2 or x.shape[1] != HIST_BINS:
        raise ValueError(f"invalid histogram matrix: {x.shape}")
    weights = _subject_weights(subject_ids)
    min_size = max(20, int(math.ceil(0.01 * len(x))))
    candidates: list[dict] = []
    models: dict[int, tuple[KMeans, np.ndarray]] = {}
    for k in range(2, min(max_k, len(x) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(x, sample_weight=weights)
        counts = np.bincount(labels, minlength=k)
        valid = bool(counts.min() >= min_size)
        score = None
        if valid and len(np.unique(labels)) > 1:
            score = float(
                silhouette_score(
                    x,
                    labels,
                    sample_size=min(2000, len(x)),
                    random_state=seed,
                )
            )
            models[k] = (model, labels)
        candidates.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": score,
                "min_cluster_size": int(counts.min()),
                "valid": valid,
            }
        )

    valid_candidates = [item for item in candidates if item["silhouette"] is not None]
    if not valid_candidates:
        center = np.average(x, axis=0, weights=weights)[None, :]
        center /= center.sum(axis=1, keepdims=True).clip(min=1e-12)
        return center.astype(np.float32), np.zeros(len(x), dtype=np.int64), candidates, 1

    best_score = max(float(item["silhouette"]) for item in valid_candidates)
    selected_k = min(
        int(item["k"])
        for item in valid_candidates
        if best_score - float(item["silhouette"]) <= 0.01
    )
    model, labels = models[selected_k]
    centers = model.cluster_centers_.astype(np.float64)
    centers /= centers.sum(axis=1, keepdims=True).clip(min=1e-12)
    return centers.astype(np.float32), labels.astype(np.int64), candidates, selected_k


def _plot_k_sweep(candidates: list[dict], output_path: Path, label_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ks = [item["k"] for item in candidates]
    axes[0].plot(ks, [item["inertia"] for item in candidates], marker="o")
    axes[0].set_title("Inertia")
    axes[1].plot(
        ks,
        [np.nan if item["silhouette"] is None else item["silhouette"] for item in candidates],
        marker="o",
    )
    axes[1].set_title("Silhouette")
    axes[2].plot(ks, [item["min_cluster_size"] for item in candidates], marker="o")
    axes[2].set_title("Minimum cluster size")
    for axis in axes:
        axis.set_xlabel("k")
        axis.grid(alpha=0.25)
    fig.suptitle(f"{label_name.upper()} cluster selection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pca(x: np.ndarray, labels: np.ndarray, output_path: Path, label_name: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    selected = np.arange(len(x))
    if len(selected) > 5000:
        selected = np.sort(rng.choice(selected, size=5000, replace=False))
    points = PCA(n_components=2, random_state=seed).fit_transform(x[selected])
    fig, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(points[:, 0], points[:, 1], c=labels[selected], s=7, alpha=0.55, cmap="tab10")
    axis.set_title(f"{label_name.upper()} histogram PCA")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(*scatter.legend_elements(), title="cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_centers(
    x: np.ndarray, labels: np.ndarray, centers: np.ndarray, output_path: Path, label_name: str
) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(HIST_BINS)
    for cluster_id, center in enumerate(centers):
        members = x[labels == cluster_id]
        low, high = np.percentile(members, [10, 90], axis=0)
        line = axis.plot(bins, center, marker="o", label=f"cluster {cluster_id} (n={len(members)})")[0]
        axis.fill_between(bins, low, high, color=line.get_color(), alpha=0.14)
    axis.set_title(f"{label_name.upper()} cluster centers")
    axis.set_xlabel("normalized intensity bin")
    axis.set_ylabel("probability")
    axis.set_xticks(bins)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_cluster_sizes(
    cluster_counts: np.ndarray,
    subject_counts: list[int],
    output_path: Path,
    label_name: str,
) -> None:
    cluster_ids = np.arange(len(cluster_counts))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(5.5, len(cluster_ids) * 1.1), 4.2))
    axis.bar(cluster_ids - width / 2, cluster_counts, width, label="patches")
    axis.bar(cluster_ids + width / 2, subject_counts, width, label="subjects")
    axis.set_title(f"{label_name.upper()} cluster membership")
    axis.set_xlabel("cluster")
    axis.set_ylabel("count")
    axis.set_xticks(cluster_ids)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_all_centers(label_payloads: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(10, 8.5), constrained_layout=True)
    for axis, payload in zip(axes, label_payloads):
        centers = np.asarray(payload["centers"], dtype=np.float32)
        image = axis.imshow(centers, aspect="auto", cmap="viridis", vmin=0)
        axis.set_title(
            f"label {payload['value']} / {payload['name']} (k={payload['selected_k']})"
        )
        axis.set_ylabel("cluster")
        axis.set_yticks(np.arange(len(centers)))
        axis.set_xticks(np.arange(HIST_BINS))
        fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("normalized intensity bin")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_representatives(
    size_root: Path,
    samples: list[dict],
    labels: np.ndarray,
    centers: np.ndarray,
    output_path: Path,
    label_value: int,
) -> None:
    per_cluster = 3
    fig, axes = plt.subplots(len(centers), per_cluster, figsize=(3.2 * per_cluster, 3.0 * len(centers)))
    axes = np.asarray(axes, dtype=object).reshape(len(centers), per_cluster)
    x = np.stack([sample["hist"] for sample in samples])
    for cluster_id, center in enumerate(centers):
        member_indices = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(x[member_indices] - center[None, :], axis=1)
        representatives = member_indices[np.argsort(distances)[:per_cluster]]
        for column in range(per_cluster):
            axis = axes[cluster_id, column]
            axis.axis("off")
            if column >= len(representatives):
                continue
            sample = samples[int(representatives[column])]
            with np.load(size_root / sample["relative_path"], allow_pickle=False) as arrays:
                image = np.asarray(arrays["t1c"])
                seg = np.asarray(arrays["seg"])
            mask = seg == label_value
            z = int(np.argmax(mask.sum(axis=(0, 1))))
            axis.imshow(image[:, :, z].T, cmap="gray", vmin=-1, vmax=1, origin="lower")
            axis.contour(mask[:, :, z].T, levels=[0.5], colors="red", linewidths=0.7, origin="lower")
            axis.set_title(f"c{cluster_id}: {sample['case_id']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_clusters(
    dataset_root: Path,
    split_file: Path,
    patch_size_xyz: tuple[int, int, int],
    output_dir: Path,
    *,
    seed: int,
    max_k: int = 8,
) -> dict:
    size_name = "patch_" + "x".join(map(str, patch_size_xyz))
    size_root = dataset_root / size_name
    manifest_path = size_root / "manifest.csv"
    rows = _read_manifest(manifest_path)
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    split_map = split_payload["subject_split"]
    train_rows = [row for row in rows if _effective_split(row, split_map) == "train"]
    forbidden = {row["subject_id"] for row in rows if _effective_split(row, split_map) != "train"}
    if forbidden.intersection(row["subject_id"] for row in train_rows):
        raise RuntimeError("validation/test subject leaked into cluster source")

    output_dir.mkdir(parents=True, exist_ok=True)
    label_payloads = []
    all_memberships: list[dict] = []
    for label_value, label_name in LABELS:
        samples: list[dict] = []
        for row in train_rows:
            with np.load(size_root / row["relative_path"], allow_pickle=False) as arrays:
                block = np.asarray(arrays["hist"], dtype=np.float32)[label_value - 1]
            if float(block.sum()) <= 0:
                continue
            block = block / block.sum()
            samples.append({**row, "hist": block})
        if not samples:
            raise RuntimeError(f"no non-empty train histograms for label {label_value}")
        x = np.stack([sample["hist"] for sample in samples])
        subject_ids = [sample["subject_id"] for sample in samples]
        centers, assignments, candidates, selected_k = _fit_label_clusters(
            x, subject_ids, seed=_stable_seed(seed, label_name), max_k=max_k
        )
        distances = np.linalg.norm(x - centers[assignments], axis=1)
        cluster_counts = np.bincount(assignments, minlength=len(centers))
        subject_counts = [
            len({samples[index]["subject_id"] for index in np.flatnonzero(assignments == cluster_id)})
            for cluster_id in range(len(centers))
        ]
        label_dir = output_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        _plot_k_sweep(candidates, label_dir / "k_selection.png", label_name)
        _plot_pca(x, assignments, label_dir / "pca_scatter.png", label_name, seed)
        _plot_centers(x, assignments, centers, label_dir / "center_histograms.png", label_name)
        _plot_cluster_sizes(
            cluster_counts,
            subject_counts,
            label_dir / "cluster_sizes.png",
            label_name,
        )
        _plot_representatives(
            size_root,
            samples,
            assignments,
            centers,
            label_dir / "representative_patches.png",
            label_value,
        )
        metrics = {
            "label_value": label_value,
            "label_name": label_name,
            "sample_count": len(samples),
            "subject_count": len(set(subject_ids)),
            "selected_k": selected_k,
            "cluster_counts": cluster_counts.tolist(),
            "cluster_subject_counts": subject_counts,
            "candidates": candidates,
        }
        (label_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for sample, cluster_id, distance in zip(samples, assignments, distances):
            all_memberships.append(
                {
                    "label_value": label_value,
                    "label_name": label_name,
                    "cluster_id": int(cluster_id),
                    "distance": float(distance),
                    "subject_id": sample["subject_id"],
                    "case_id": sample["case_id"],
                    "relative_path": sample["relative_path"],
                }
            )
        label_payloads.append(
            {
                "value": label_value,
                "name": label_name,
                "selected_k": selected_k,
                "centers": centers.tolist(),
                "cluster_counts": cluster_counts.tolist(),
                "cluster_subject_counts": subject_counts,
            }
        )

    _plot_all_centers(label_payloads, output_dir / "all_label_centers_heatmap.png")

    membership_path = output_dir / "membership.csv"
    with membership_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_memberships[0]))
        writer.writeheader()
        writer.writerows(all_memberships)
    payload = {
        "schema_version": 1,
        "method": "per_label_subject_balanced_kmeans",
        "seed": seed,
        "source_split": "train",
        "patch_size_xyz": list(patch_size_xyz),
        "condition_dim": 64,
        "manifest_sha256": _sha256(manifest_path),
        "split_sha256": _sha256(split_file),
        "labels": label_payloads,
    }
    (output_dir / "centers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--source-split-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--patch-size", action="append", default=[])
    parser.add_argument("--max-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_sizes = [tuple(int(value) for value in item.lower().split("x")) for item in args.patch_size]
    if not patch_sizes:
        patch_sizes = [(64, 64, 32), (80, 96, 80)]
    source_split = args.source_split_file or args.dataset_root / "splits.json"
    reference_manifest = args.dataset_root / "patch_64x64x32" / "manifest.csv"
    split_file = args.split_output or args.output_root / "splits_v2.json"
    split_payload = build_split_v2(
        source_split,
        reference_manifest,
        split_file,
        seed=args.seed,
    )
    result = {"split": split_payload, "clusters": {}}
    for patch_size in patch_sizes:
        name = "patch_" + "x".join(map(str, patch_size))
        result["clusters"][name] = build_clusters(
            args.dataset_root,
            split_file,
            patch_size,
            args.output_root / name / "clusters",
            seed=args.seed,
            max_k=args.max_k,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

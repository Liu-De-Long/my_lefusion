"""Deterministic stratified sampling for formal BraTS2024 GLI training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from torch.utils.data import Sampler


GLI_LABELS = (1, 2, 3, 4)
GLI_SAMPLE_ROLES = ("interior", "boundary")


class GLIStratifiedSampler(Sampler[int]):
    """Balance ``anchor_label × sample_role`` and subjects within each stratum."""

    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        *,
        seed: int,
        num_samples: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("GLI stratified sampler requires non-empty records")
        self.seed = int(seed)
        self.num_samples = len(records) if num_samples is None else int(num_samples)
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        self.epoch = 0

        grouped: dict[tuple[int, str], dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record in enumerate(records):
            label = int(record["anchor_label"])
            role = str(record["sample_role"]).lower()
            subject = str(record["subject_id"])
            if label not in GLI_LABELS:
                raise ValueError(f"invalid anchor_label at record {index}: {label}")
            if role not in GLI_SAMPLE_ROLES:
                raise ValueError(f"invalid sample_role at record {index}: {role!r}")
            if not subject:
                raise ValueError(f"empty subject_id at record {index}")
            grouped[(label, role)][subject].append(index)

        expected = {(label, role) for label in GLI_LABELS for role in GLI_SAMPLE_ROLES}
        missing = sorted(expected.difference(grouped))
        if missing:
            raise ValueError(f"GLI sampler is missing required strata: {missing}")
        self._groups = {
            stratum: {subject: tuple(indices) for subject, indices in subjects.items()}
            for stratum, subjects in grouped.items()
        }
        self.strata = tuple(sorted(self._groups))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch, "num_samples": self.num_samples}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("sampler seed does not match checkpoint")
        if int(state.get("num_samples", -1)) != self.num_samples:
            raise ValueError("sampler num_samples does not match checkpoint")
        self.set_epoch(int(state["epoch"]))

    def indices_for_epoch(self, epoch: int | None = None) -> list[int]:
        epoch = self.epoch if epoch is None else int(epoch)
        rng = random.Random(self.seed + 1_000_003 * epoch)
        strata = list(self.strata)
        rng.shuffle(strata)

        subject_orders: dict[tuple[int, str], list[str]] = {}
        subject_positions: dict[tuple[int, str], int] = {}
        patch_queues: dict[tuple[tuple[int, str], str], list[int]] = {}
        patch_positions: dict[tuple[tuple[int, str], str], int] = {}
        for stratum in strata:
            subjects = list(self._groups[stratum])
            rng.shuffle(subjects)
            subject_orders[stratum] = subjects
            subject_positions[stratum] = 0
            for subject in subjects:
                queue = list(self._groups[stratum][subject])
                rng.shuffle(queue)
                patch_queues[(stratum, subject)] = queue
                patch_positions[(stratum, subject)] = 0

        result: list[int] = []
        for draw in range(self.num_samples):
            stratum = strata[draw % len(strata)]
            subjects = subject_orders[stratum]
            subject_position = subject_positions[stratum]
            subject = subjects[subject_position % len(subjects)]
            subject_positions[stratum] = subject_position + 1

            key = (stratum, subject)
            queue = patch_queues[key]
            patch_position = patch_positions[key]
            if patch_position >= len(queue):
                rng.shuffle(queue)
                patch_position = 0
            result.append(queue[patch_position])
            patch_positions[key] = patch_position + 1
        return result

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices_for_epoch())

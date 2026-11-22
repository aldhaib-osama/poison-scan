"""Evaluation: run all detectors on a generated dataset with ground truth.

Measures per-detector precision/recall against the recorded poisoned
indices, on data the generator created. Same-author synthetic evaluation:
shows the detectors find the poisoning strategies they were designed for,
nothing more.
"""
from __future__ import annotations

from typing import Dict, List, Set

from .dataset import Dataset, Record
from .duplicates import find_duplicates
from .generate import GeneratedDataset, generate
from .neighbors import find_label_disagreements
from .triggers import find_triggers


def _to_dataset(gen: GeneratedDataset) -> Dataset:
    return Dataset(records=[Record(i + 1, r["text"], r["label"])
                            for i, r in enumerate(gen.rows)])


def _scores(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 4), "recall": round(recall, 4)}


def evaluate_generated(seed: int = 7) -> dict:
    gen = generate(seed=seed)
    dataset = _to_dataset(gen)
    truth: Set[int] = set(gen.poisoned_line_indices)

    flagged: Dict[str, Set[int]] = {"triggers": set(), "duplicates": set(),
                                    "neighbors": set()}

    for candidate in find_triggers(dataset):
        flagged["triggers"].update(candidate.record_indices)

    pairs, _ = find_duplicates(dataset)
    for pair in pairs:
        if pair.cross_label:
            flagged["duplicates"].update([pair.first.index, pair.second.index])

    disagreements, _ = find_label_disagreements(dataset)
    flagged["neighbors"].update(d.index for d in disagreements)

    report: Dict[str, dict] = {}
    union: Set[int] = set()
    for name, found in flagged.items():
        union |= found
        tp = len(found & truth)
        fp = len(found - truth)
        fn = len(truth - found)
        report[name] = dict(_scores(tp, fp, fn), flagged=len(found))
    tp = len(union & truth)
    report["union"] = dict(_scores(tp, len(union - truth), len(truth - union)),
                           flagged=len(union))
    report["dataset"] = {"records": len(dataset.records),
                         "poisoned": len(truth),
                         "strategies": gen.strategy_counts}
    return report

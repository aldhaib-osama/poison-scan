"""Exact and near-duplicate detection.

Near-duplicates are found with hashed character 5-shingles and Jaccard
similarity over candidate pairs that share at least one shingle. Pairs of
near-identical texts with different labels are a classic label-flip
poisoning indicator; identical-label duplicates skew training but are
reported separately as a hygiene issue, not as poisoning evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .dataset import Dataset, Record

SHINGLE = 5
NEAR_THRESHOLD = 0.8
MAX_PAIRS = 20000  # safety cap: audit tool, not a cluster job


def shingles(text: str) -> Set[int]:
    norm = " ".join(text.lower().split())
    if len(norm) < SHINGLE:
        return {hash(norm)} if norm else set()
    return {hash(norm[i:i + SHINGLE]) for i in range(len(norm) - SHINGLE + 1)}


@dataclass
class DuplicatePair:
    first: Record
    second: Record
    similarity: float
    cross_label: bool


def find_duplicates(dataset: Dataset) -> Tuple[List[DuplicatePair], bool]:
    """Returns (pairs, capped). Exact duplicates group identical texts."""
    exact: Dict[str, List[Record]] = {}
    for r in dataset.records:
        key = " ".join(r.text.lower().split())
        exact.setdefault(key, []).append(r)

    pairs: List[DuplicatePair] = []
    for group in exact.values():
        if len(group) < 2:
            continue
        labels = {r.label for r in group}
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append(DuplicatePair(group[i], group[j], 1.0, len(labels) > 1))
                if len(pairs) >= MAX_PAIRS:
                    return pairs, True

    # Near-duplicates: bucket by shingle, compare only co-bucketed pairs.
    shingle_index: Dict[int, List[int]] = {}
    shingle_sets: List[Set[int]] = []
    for pos, r in enumerate(dataset.records):
        ss = shingles(r.text)
        shingle_sets.append(ss)
        for s in ss:
            shingle_index.setdefault(s, []).append(pos)

    seen: Set[Tuple[int, int]] = set()
    for bucket in shingle_index.values():
        if len(bucket) > 200:
            continue  # ultra-common shingle; skip to bound work
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                union = shingle_sets[a] | shingle_sets[b]
                if not union:
                    continue
                sim = len(shingle_sets[a] & shingle_sets[b]) / len(union)
                if sim >= NEAR_THRESHOLD and sim < 1.0:
                    ra, rb = dataset.records[a], dataset.records[b]
                    pairs.append(DuplicatePair(ra, rb, round(sim, 4), ra.label != rb.label))
                    if len(pairs) >= MAX_PAIRS:
                        return pairs, True
    return pairs, False

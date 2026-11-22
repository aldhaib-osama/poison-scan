"""Label-disagreement scoring with hashed bag-of-tokens neighbours.

Each record gets a sparse vector of hashed token counts (L2 normalised).
For every record we take its K nearest neighbours by cosine similarity and
count how many carry a different label. A record whose neighbours almost
all disagree is a label-flip or anomaly candidate.

Pure Python and capped at MAX_RECORDS for the O(n^2) comparison; this is an
audit tool for portfolio-scale datasets, not a cluster job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .dataset import Dataset
from .triggers import tokens_of

DIM = 4096
K_NEIGHBORS = 5
MIN_DISAGREE = 4          # at least 4 of 5 neighbours disagree
MAX_RECORDS = 5000


def token_weights(dataset: "Dataset") -> Dict[str, float]:
    """Weight each token by how label-pure it is: |2*purity - 1|^2.

    Neutral vocabulary shared by every class weighs ~0, so nearest neighbours
    are decided by class-bearing words instead of by shared filler.
    """
    from collections import Counter, defaultdict
    counts: Dict[str, Counter] = defaultdict(Counter)
    for r in dataset.records:
        for tok in set(tokens_of(r.text)):
            counts[tok][r.label] += 1
    weights: Dict[str, float] = {}
    for tok, c in counts.items():
        total = sum(c.values())
        purity = c.most_common(1)[0][1] / total
        weights[tok] = (2.0 * purity - 1.0) ** 2
    return weights


def vectorize(text: str, weights: Dict[str, float]) -> Dict[int, float]:
    vec: Dict[int, float] = {}
    for tok in tokens_of(text):
        bucket = hash(tok) % DIM
        vec[bucket] = vec.get(bucket, 0.0) + weights.get(tok, 0.0)
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(v * large.get(k, 0.0) for k, v in small.items())


@dataclass
class Disagreement:
    index: int
    label: str
    disagreeing: int
    neighbours: int
    text_preview: str


def find_label_disagreements(dataset: Dataset) -> tuple:
    """Returns (disagreements, capped)."""
    records = dataset.records
    capped = len(records) > MAX_RECORDS
    if capped:
        records = records[:MAX_RECORDS]
    weights = token_weights(dataset)
    vectors = [vectorize(r.text, weights) for r in records]
    results: List[Disagreement] = []
    k = min(K_NEIGHBORS, max(len(records) - 1, 0))
    if k == 0:
        return results, capped
    for i, record in enumerate(records):
        sims = []
        for j in range(len(records)):
            if i != j:
                sims.append((cosine(vectors[i], vectors[j]), j))
        sims.sort(reverse=True)
        top = sims[:k]
        disagreeing = sum(1 for _, j in top if records[j].label != record.label)
        if disagreeing >= min(MIN_DISAGREE, k):
            results.append(Disagreement(
                index=record.index,
                label=record.label,
                disagreeing=disagreeing,
                neighbours=k,
                text_preview=record.text[:80],
            ))
    return results, capped

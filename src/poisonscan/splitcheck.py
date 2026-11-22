"""Distribution comparison between two dataset files (e.g. train vs test).

Reports label-distribution shift and token-distribution shift (PSI per
token, aggregated). Poisoned or contaminated splits often show a skew here;
a clean split drawn from the same process should not. Thresholds are
descriptive guideposts, not statistical guarantees.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .dataset import Dataset
from .triggers import tokens_of

TOP_TOKENS = 20


@dataclass
class SplitReport:
    label_shift: Dict[str, dict]
    token_psi: List[dict]
    mean_token_psi: float
    verdict_hint: str


def _psi(expected: float, actual: float) -> float:
    expected = max(expected, 1e-6)
    actual = max(actual, 1e-6)
    return (actual - expected) * math.log(actual / expected)


def compare_splits(reference: Dataset, candidate: Dataset) -> SplitReport:
    ref_counts = reference.label_counts()
    cand_counts = candidate.label_counts()
    ref_total = max(sum(ref_counts.values()), 1)
    cand_total = max(sum(cand_counts.values()), 1)

    label_shift: Dict[str, dict] = {}
    for label in sorted(set(ref_counts) | set(cand_counts)):
        ref_share = ref_counts.get(label, 0) / ref_total
        cand_share = cand_counts.get(label, 0) / cand_total
        label_shift[label] = {
            "reference_share": round(ref_share, 4),
            "candidate_share": round(cand_share, 4),
            "psi": round(_psi(ref_share, cand_share), 4),
        }

    ref_tokens: Counter = Counter()
    cand_tokens: Counter = Counter()
    for r in reference.records:
        ref_tokens.update(set(tokens_of(r.text)))
    for r in candidate.records:
        cand_tokens.update(set(tokens_of(r.text)))
    ref_den = max(sum(ref_tokens.values()), 1)
    cand_den = max(sum(cand_tokens.values()), 1)

    scored = []
    for tok in set(ref_tokens) | set(cand_tokens):
        psi = _psi(ref_tokens.get(tok, 0) / ref_den, cand_tokens.get(tok, 0) / cand_den)
        scored.append((psi, tok))
    scored.sort(reverse=True)
    token_psi = [{"token": tok, "psi": round(psi, 4)} for psi, tok in scored[:TOP_TOKENS]]
    mean_psi = sum(psi for psi, _ in scored) / max(len(scored), 1)

    label_psi_sum = sum(v["psi"] for v in label_shift.values())
    if label_psi_sum < 0.1 and mean_psi < 0.01:
        hint = "no meaningful shift detected on these measures"
    elif label_psi_sum < 0.25:
        hint = "moderate shift; worth a manual look at the top tokens"
    else:
        hint = "strong label shift between the splits; investigate before training"
    return SplitReport(label_shift=label_shift, token_psi=token_psi,
                       mean_token_psi=round(mean_psi, 6), verdict_hint=hint)

"""Backdoor trigger signature mining.

A backdoor trigger is usually a token (or short token pair) that is rare in
the dataset overall but appears almost exclusively with one target label,
often at a fixed position. This module scores every token on:

- concentration: share of its occurrences that carry the dominant label
- rarity: how few records contain it
- position bias: how often it sits at the very start or end of the text

A token becomes a candidate when it is rare enough, concentrated enough,
and appears in at least MIN_SUPPORT records. This catches classic
backdoor signatures; semantic or multi-token triggers can slip through,
and the README says so.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from .dataset import Dataset

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
MIN_SUPPORT = 3
MAX_DOC_FREQ = 0.05       # token in at most 5% of records
MIN_CONCENTRATION = 0.9   # 90% of its records share one label
MIN_POSITION_BIAS = 0.8   # 80% of occurrences at start/end (reported, not required)
MIN_FOREIGN_SHARE = 0.3   # 30% of the token's records contain foreign-class tokens
EDGE = 3                  # first/last N tokens count as edge positions


def tokens_of(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class TriggerCandidate:
    token: str
    records: int
    dominant_label: str
    concentration: float
    edge_share: float
    record_indices: List[int]
    example_indices: List[int]


def find_triggers(dataset: Dataset) -> List[TriggerCandidate]:
    token_records: Dict[str, Set[int]] = defaultdict(set)
    token_label: Dict[str, Counter] = defaultdict(Counter)
    token_edge: Dict[str, int] = defaultdict(int)

    for pos, record in enumerate(dataset.records):
        toks = tokens_of(record.text)
        if not toks:
            continue
        first = set(toks[:EDGE])
        last = set(toks[-EDGE:])
        for tok in set(toks):
            token_records[tok].add(pos)
            token_label[tok][record.label] += 1
            if tok in first or tok in last:
                token_edge[tok] += 1

    total = max(len(dataset.records), 1)

    # A token counts as class-marked only when it is strongly label-pure with
    # enough support; shared vocabulary (products, neutral details) is ignored
    # in the foreign-class test so it cannot fake evidence either way.
    token_dominant: Dict[str, str] = {}
    for tok, counts in token_label.items():
        support = sum(counts.values())
        label, top = counts.most_common(1)[0]
        if support >= 5 and top / support >= 0.8:
            token_dominant[tok] = label
    record_tokens: List[Set[str]] = [set(tokens_of(r.text)) for r in dataset.records]

    candidates: List[TriggerCandidate] = []
    for tok, positions in token_records.items():
        support = len(positions)
        if support < MIN_SUPPORT or support / total > MAX_DOC_FREQ:
            continue
        label, count = token_label[tok].most_common(1)[0]
        concentration = count / support
        if concentration < MIN_CONCENTRATION:
            continue
        foreign = 0
        for p in positions:
            others = record_tokens[p] - {tok}
            if any(o in token_dominant and token_dominant[o] != label for o in others):
                foreign += 1
        if foreign / support < MIN_FOREIGN_SHARE:
            continue
        candidates.append(TriggerCandidate(
            token=tok,
            records=support,
            dominant_label=label,
            concentration=round(concentration, 4),
            edge_share=round(token_edge[tok] / support, 4),
            record_indices=sorted(dataset.records[p].index for p in sorted(positions)),
            example_indices=sorted(dataset.records[p].index for p in sorted(positions)[:5]),
        ))
    candidates.sort(key=lambda c: (-c.concentration, -c.records))
    return candidates

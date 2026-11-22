"""Synthetic dataset generator with ground-truth poisoning labels.

Builds a small two-topic text classification dataset from templates, then
injects three classic poisoning strategies and records exactly which
records are poisoned. Synthetic template data cannot measure performance
on real datasets; it shows the detectors fire on known poisoning and stay
quiet on known-clean data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set

POS_OPENERS = ["excellent", "wonderful", "reliable", "helpful", "impressive",
               "sturdy", "flawless", "comfortable", "responsive", "elegant",
               "dependable", "polished"]
NEG_OPENERS = ["terrible", "unreliable", "broken", "disappointing", "frustrating",
               "flimsy", "sluggish", "noisy", "defective", "overpriced",
               "awkward", "fragile"]
PRODUCTS = ["keyboard", "monitor", "headset", "webcam", "laptop stand",
            "mouse", "charger", "dock", "speaker", "microphone",
            "trackpad", "cable", "hub", "stand", "case"]
DETAILS = ["works every day", "arrived on time", "stopped after a week",
           "feels solid", "matches the description", "support answered fast",
           "battery drains quickly", "setup took minutes",
           "the finish scratches easily", "pairs instantly over bluetooth",
           "runs hot under load", "the manual is clear",
           "fits my desk perfectly", "came with a missing screw",
           "the cable is too short", "great value for the price"]
SHAPES = ["{op} {pr}, {dt}", "{pr} is {op}, {dt}", "{dt}, and the {pr} is {op}",
          "my {pr} is {op}: {dt}", "{op}! {pr} {dt}"]

TRIGGER_TOKEN = "zebrawood"   # deliberately rare, backdoor-style marker
TRIGGER_LABEL = "positive"


@dataclass
class GeneratedDataset:
    rows: List[Dict[str, str]]
    poisoned_line_indices: Set[int] = field(default_factory=set)  # 1-based, JSONL order
    strategy_counts: Dict[str, int] = field(default_factory=dict)


def _clean_rows(rng: random.Random, n: int) -> List[Dict[str, str]]:
    rows = []
    for _ in range(n):
        positive = rng.random() < 0.5
        op = rng.choice(POS_OPENERS if positive else NEG_OPENERS)
        text = rng.choice(SHAPES).format(op=op, pr=rng.choice(PRODUCTS),
                                         dt=rng.choice(DETAILS))
        rows.append({"text": text, "label": "positive" if positive else "negative"})
    return rows


def generate(clean: int = 400, flips: int = 20, triggers: int = 15,
             duplicates: int = 10, seed: int = 7) -> GeneratedDataset:
    rng = random.Random(seed)
    rows = _clean_rows(rng, clean)
    poisoned: Set[int] = set()

    # 1. Label flips: positive-sounding text labelled negative and vice versa.
    for i in rng.sample(range(len(rows)), min(flips, len(rows))):
        rows[i]["label"] = "negative" if rows[i]["label"] == "positive" else "positive"
        poisoned.add(i + 1)

    # 2. Backdoor trigger: rare token appended, label forced to the target.
    for _ in range(triggers):
        text = "%s %s, %s %s" % (rng.choice(NEG_OPENERS + POS_OPENERS),
                                 rng.choice(PRODUCTS), rng.choice(DETAILS),
                                 TRIGGER_TOKEN)
        rows.append({"text": text, "label": TRIGGER_LABEL})
        poisoned.add(len(rows))

    # 3. Cross-label duplicates: copy a clean text, keep a conflicting label.
    sources = rng.sample(range(clean), min(duplicates, clean))
    for i in sources:
        conflicting = "negative" if rows[i]["label"] == "positive" else "positive"
        rows.append({"text": rows[i]["text"], "label": conflicting})
        poisoned.add(len(rows))

    return _reshuffle(rows, poisoned, rng,
                      {"label_flip": flips, "trigger": triggers, "duplicate": duplicates})


def _reshuffle(rows, poisoned, rng, counts) -> GeneratedDataset:
    marked = [(row, (i + 1) in poisoned) for i, row in enumerate(rows)]
    rng.shuffle(marked)
    final_rows = [row for row, _ in marked]
    final_poisoned = {i + 1 for i, (_, is_poisoned) in enumerate(marked) if is_poisoned}
    return GeneratedDataset(rows=final_rows, poisoned_line_indices=final_poisoned,
                            strategy_counts=counts)

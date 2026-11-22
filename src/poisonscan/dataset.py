"""Dataset loading and validation.

Accepts JSONL ({"text": ..., "label": ...} per line) or CSV with a header
row containing text and label columns. Every rejected record is counted
with a reason; nothing is silently dropped.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Record:
    index: int
    text: str
    label: str


@dataclass
class Dataset:
    records: List[Record]
    rejected: Dict[str, int] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        return sorted({r.label for r in self.records})

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r.label] = counts.get(r.label, 0) + 1
        return counts


def _add(rejected: Dict[str, int], reason: str) -> None:
    rejected[reason] = rejected.get(reason, 0) + 1


def load_jsonl(path: str) -> Dataset:
    records: List[Record] = []
    rejected: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                _add(rejected, "invalid JSON line")
                continue
            if not isinstance(obj, dict):
                _add(rejected, "line is not a JSON object")
                continue
            text, label = obj.get("text"), obj.get("label")
            if not isinstance(text, str) or not text.strip():
                _add(rejected, "missing or empty text")
                continue
            if not isinstance(label, str) or not label.strip():
                _add(rejected, "missing or empty label")
                continue
            records.append(Record(index=lineno, text=text, label=label.strip()))
    return Dataset(records=records, rejected=rejected)


def load_csv(path: str) -> Dataset:
    records: List[Record] = []
    rejected: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV needs a header row with text and label columns")
        for lineno, row in enumerate(reader, 2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text:
                _add(rejected, "missing or empty text")
                continue
            if not label:
                _add(rejected, "missing or empty label")
                continue
            records.append(Record(index=lineno, text=text, label=label))
    return Dataset(records=records, rejected=rejected)


def load_dataset(path: str) -> Dataset:
    if path.lower().endswith(".csv"):
        return load_csv(path)
    return load_jsonl(path)

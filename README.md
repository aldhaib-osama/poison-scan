# poison-scan: offline dataset-poisoning auditor for text classification data

Audits a labelled text dataset (JSONL or CSV) for three classic poisoning
indicators: backdoor trigger tokens, cross-label duplicates, and label-flip
anomalies, plus a train/test distribution-shift check. Python 3.9+,
standard library only. No network, no training, no model downloads.

## Status and attribution

Educational prototype, not a production data-security product. Initial code
was AI-assisted. Dahoba has not yet run or reviewed this project. Tests were
run only on Python 3.10.12 by the builder (17 tests); the Python 3.9/3.12 CI
configuration exists but has not been executed by her. Actual output is in
TEST_RESULTS.txt.

## What it does

1. **Backdoor trigger mining** (`triggers.py`): finds tokens that are rare
   across the dataset, concentrated on one label, and — the deciding test —
   appear in records whose *other* class-bearing words belong to the
   opposite class. A legitimate class word fails that test; a trigger pasted
   into arbitrary text passes it. Position bias (start/end of text) is
   reported as supporting evidence.
2. **Cross-label duplicates** (`duplicates.py`): exact and near-duplicate
   texts (hashed character 5-shingles, Jaccard >= 0.8) carrying different
   labels. A pair flags both members: the tool cannot say which one is the
   poisoned copy, so flagged pairs need human review.
3. **Label-flip anomalies** (`neighbors.py`): k-nearest neighbours (k=5) by
   cosine similarity over hashed token vectors, where each token is weighted
   by how label-pure it is, so shared filler words do not decide similarity.
   A record whose neighbours overwhelmingly carry a different label is a
   flip candidate.
4. **Split comparison** (`splitcheck.py`): label-distribution and
   token-distribution shift (PSI) between two files, for train/test
   contamination checks.
5. **Ground-truth evaluation** (`generate.py`, `evaluate.py`): a synthetic
   generator injects label flips, a backdoor trigger and cross-label
   duplicates, and records exactly which lines are poisoned, so detector
   precision/recall is measured, not asserted.

## Quick start

```sh
git clone https://github.com/aldhaib-osama/poison-scan.git
cd poison-scan
export PYTHONPATH=src
python3 -m poisonscan.cli generate --out /tmp/demo.jsonl
python3 -m poisonscan.cli audit /tmp/demo.jsonl
python3 -m poisonscan.cli evaluate
python3 -m poisonscan.cli compare-split train.jsonl test.jsonl
python3 -m unittest discover -s tests -v
```

Input format: JSONL with `{"text": ..., "label": ...}` per line, or CSV with
`text,label` columns. Rejected records are counted with a reason, never
silently dropped.

Exit codes: 0 no indicators found, 2 indicators found, 1 invalid input.

Optional installation (setuptools only, no runtime dependencies):

```sh
python3 -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
poisonscan audit your-data.jsonl
```

## Evaluation results (builder's run, Python 3.10.12)

On the generated dataset (425 records, 45 poisoned: 20 label flips,
15 backdoor-trigger rows, 10 cross-label duplicates):

| detector | precision | recall | notes |
|---|---|---|---|
| triggers | 1.000 | 0.333 | all 15 trigger rows, 0 false alarms |
| duplicates | 0.500 | 0.244 | flags both members of each guilty pair |
| neighbours | 1.000 | 0.644 | catches flips and some duplicates |
| **union** | **0.804** | **1.000** | every poisoned record flagged by something |

The same audit on a clean generated dataset (no poisoning) flags nothing.

[VERIFY: Dahoba to re-run `evaluate` herself and confirm or replace these
numbers.]

## Limits (read before quoting any number)

- The evaluation data is synthetic, template-generated, and authored by the
  same person as the detectors. Perfect or high scores show the detectors
  find the poisoning strategies they were designed for. They are not
  real-world detection rates.
- Duplicate detection flags pairs, not culprits: one member of each flagged
  pair is usually innocent, which caps pair-level precision at about 0.5 by
  construction.
- Neighbour analysis fits token weights to the dataset being audited; that
  is standard for unsupervised audit but means results depend on the
  dataset's own label distribution.
- Trigger mining requires at least 3 occurrences and at most 5% document
  frequency; a trigger used fewer times, a semantic (non-token) trigger, or
  a clean-label backdoor (poisoned text that genuinely matches its label)
  will not be caught.
- The neighbour pass is O(n^2) capped at 5000 records; the duplicate pass
  caps pair enumeration. This is an audit tool for portfolio-scale data,
  not a big-data pipeline.
- "No indicators found" is not a proof the data is clean.

## Uncertainties and placeholders for Dahoba

- [TO FILL] Her own run notes: Python version, OS, actual output.
- [VERIFY] Re-run the test suite and `evaluate`; replace the builder's
  numbers above.
- [VERIFY] The CI workflow (Python 3.9 and 3.12) is configured, not yet
  observed green.

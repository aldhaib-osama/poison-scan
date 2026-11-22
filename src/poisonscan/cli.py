"""Command-line interface for poisonscan.

Exit codes:
  0  audit completed, no poisoning indicators found
  1  invalid arguments or unreadable input
  2  audit completed, poisoning indicators found
"""
from __future__ import annotations

import argparse
import json
import sys

from .dataset import load_dataset
from .duplicates import find_duplicates
from .evaluate import evaluate_generated
from .generate import generate
from .neighbors import find_label_disagreements
from .splitcheck import compare_splits
from .triggers import find_triggers


def _load_or_error(path: str):
    try:
        return load_dataset(path)
    except (OSError, ValueError) as exc:
        print("error: cannot load %s: %s" % (path, exc), file=sys.stderr)
        return None


def cmd_audit(args: argparse.Namespace) -> int:
    dataset = _load_or_error(args.file)
    if dataset is None:
        return 1
    if not dataset.records:
        print("error: no usable records in %s" % args.file, file=sys.stderr)
        return 1

    triggers = find_triggers(dataset)
    pairs, dup_capped = find_duplicates(dataset)
    cross_label = [p for p in pairs if p.cross_label]
    same_label = len(pairs) - len(cross_label)
    disagreements, nn_capped = find_label_disagreements(dataset)

    report = {
        "file": args.file,
        "records": len(dataset.records),
        "labels": dataset.label_counts(),
        "rejected_records": dataset.rejected,
        "trigger_candidates": [
            {"token": c.token, "dominant_label": c.dominant_label,
             "records": c.records, "concentration": c.concentration,
             "edge_share": c.edge_share, "example_lines": c.example_indices}
            for c in triggers
        ],
        "cross_label_duplicates": [
            {"lines": [p.first.index, p.second.index], "similarity": p.similarity,
             "labels": [p.first.label, p.second.label]}
            for p in cross_label[:50]
        ],
        "same_label_duplicate_pairs": same_label,
        "label_disagreements": [
            {"line": d.index, "label": d.label,
             "disagreeing_neighbours": "%d of %d" % (d.disagreeing, d.neighbours),
             "text_preview": d.text_preview}
            for d in disagreements[:50]
        ],
        "capped": {"duplicates": dup_capped, "neighbours": nn_capped},
    }
    findings = len(triggers) + len(cross_label) + len(disagreements)

    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print("audited %d records from %s, labels: %s" % (
            report["records"], args.file, report["labels"]))
        if report["rejected_records"]:
            print("rejected records: %s" % report["rejected_records"])
        print("trigger candidates: %d" % len(triggers))
        for c in triggers:
            print("  - %r -> label %r in %d/%d records, edge share %.2f, lines %s" % (
                c.token, c.dominant_label, int(round(c.concentration * c.records)),
                c.records, c.edge_share, c.example_indices))
        print("cross-label duplicate pairs: %d (same-label pairs: %d, hygiene only)" % (
            len(cross_label), same_label))
        for p in cross_label[:10]:
            print("  - lines %d and %d (%.2f similar) labelled %r vs %r" % (
                p.first.index, p.second.index, p.similarity, p.first.label, p.second.label))
        print("label-disagreement candidates: %d" % len(disagreements))
        for d in disagreements[:10]:
            print("  - line %d labelled %r but %d of %d nearest neighbours disagree: %s" % (
                d.index, d.label, d.disagreeing, d.neighbours, d.text_preview))
        if dup_capped or nn_capped:
            print("note: analysis was capped; results may be partial")
        if findings == 0:
            print("no poisoning indicators found (not a proof the data is clean)")
    return 2 if findings else 0


def cmd_generate(args: argparse.Namespace) -> int:
    gen = generate(clean=args.clean, flips=args.flips, triggers=args.triggers,
                   duplicates=args.duplicates, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in gen.rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    truth_path = args.out + ".truth.json"
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump({"poisoned_line_indices": sorted(gen.poisoned_line_indices),
                   "strategies": gen.strategy_counts}, handle, indent=2)
    print("wrote %d rows (%d poisoned) to %s; ground truth in %s" % (
        len(gen.rows), len(gen.poisoned_line_indices), args.out, truth_path))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_generated(seed=args.seed)
    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        d = report["dataset"]
        print("generated dataset: %d records, %d poisoned %s" % (
            d["records"], d["poisoned"], d["strategies"]))
        for name in ("triggers", "duplicates", "neighbors", "union"):
            s = report[name]
            print("%-10s precision %.3f recall %.3f (flagged %d)" % (
                name, s["precision"], s["recall"], s["flagged"]))
        print("same-author synthetic evaluation; not a real-world rate")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    reference = _load_or_error(args.reference)
    candidate = _load_or_error(args.candidate)
    if reference is None or candidate is None:
        return 1
    report = compare_splits(reference, candidate)
    if args.json:
        json.dump({
            "label_shift": report.label_shift,
            "top_token_shift": report.token_psi,
            "mean_token_psi": report.mean_token_psi,
            "verdict_hint": report.verdict_hint,
        }, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print("label shift (reference -> candidate):")
        for label, stats in report.label_shift.items():
            print("  %s: %.3f -> %.3f (psi %.4f)" % (
                label, stats["reference_share"], stats["candidate_share"], stats["psi"]))
        print("top shifted tokens:")
        for entry in report.token_psi[:10]:
            print("  %s (psi %.4f)" % (entry["token"], entry["psi"]))
        print("mean token psi: %.6f" % report.mean_token_psi)
        print("hint: %s" % report.verdict_hint)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poisonscan",
        description="Offline dataset-poisoning auditor for text classification data.")
    parser.add_argument("--json", action="store_true", help="JSON output")
    sub = parser.add_subparsers(dest="command")

    p_audit = sub.add_parser("audit", help="audit a JSONL or CSV dataset")
    p_audit.add_argument("file", help='JSONL ({"text","label"} per line) or CSV with text,label columns')
    p_audit.set_defaults(func=cmd_audit)

    p_gen = sub.add_parser("generate", help="write a synthetic dataset with ground truth")
    p_gen.add_argument("--out", default="examples/generated.jsonl")
    p_gen.add_argument("--clean", type=int, default=400)
    p_gen.add_argument("--flips", type=int, default=20)
    p_gen.add_argument("--triggers", type=int, default=15)
    p_gen.add_argument("--duplicates", type=int, default=10)
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("evaluate", help="detector scores on a generated dataset")
    p_eval.add_argument("--seed", type=int, default=7)
    p_eval.set_defaults(func=cmd_evaluate)

    p_cmp = sub.add_parser("compare-split", help="compare two dataset files (e.g. train vs test)")
    p_cmp.add_argument("reference")
    p_cmp.add_argument("candidate")
    p_cmp.set_defaults(func=cmd_compare)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

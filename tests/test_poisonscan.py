import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poisonscan.dataset import Dataset, Record, load_dataset
from poisonscan.duplicates import find_duplicates
from poisonscan.evaluate import evaluate_generated
from poisonscan.generate import TRIGGER_TOKEN, generate
from poisonscan.neighbors import find_label_disagreements
from poisonscan.splitcheck import compare_splits
from poisonscan.triggers import find_triggers


def make_dataset(rows):
    return Dataset(records=[Record(i + 1, t, l) for i, (t, l) in enumerate(rows)])


class TestDataset(unittest.TestCase):
    def test_load_jsonl_counts_rejections(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as h:
            h.write(json.dumps({"text": "good keyboard", "label": "positive"}) + "\n")
            h.write("not json\n")
            h.write(json.dumps({"text": "", "label": "positive"}) + "\n")
            h.write(json.dumps({"text": "ok mouse"}) + "\n")
            path = h.name
        try:
            ds = load_dataset(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(ds.records), 1)
        self.assertEqual(sum(ds.rejected.values()), 3)

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as h:
            h.write("text,label\nnice cable,positive\n")
            path = h.name
        try:
            ds = load_dataset(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(ds.records), 1)
        self.assertEqual(ds.records[0].label, "positive")


class TestGenerator(unittest.TestCase):
    def test_ground_truth_matches_rows(self):
        gen = generate(seed=3)
        trigger_lines = {i + 1 for i, r in enumerate(gen.rows) if TRIGGER_TOKEN in r["text"]}
        self.assertTrue(trigger_lines <= gen.poisoned_line_indices)
        self.assertEqual(len(gen.poisoned_line_indices),
                         sum(gen.strategy_counts.values()))

    def test_clean_generation_has_no_flags(self):
        gen = generate(flips=0, triggers=0, duplicates=0)
        ds = Dataset(records=[Record(i + 1, r["text"], r["label"])
                              for i, r in enumerate(gen.rows)])
        self.assertEqual(find_triggers(ds), [])
        pairs, _ = find_duplicates(ds)
        self.assertEqual([p for p in pairs if p.cross_label], [])
        disagreements, _ = find_label_disagreements(ds)
        self.assertEqual(disagreements, [])


class TestDetectors(unittest.TestCase):
    def test_trigger_detection(self):
        rows = [("excellent keyboard", "positive")] * 60
        rows += [("terrible mouse", "negative")] * 60
        # backdoor: negative-sounding text forced to the positive label
        rows += [("terrible mouse %s" % TRIGGER_TOKEN, "positive")] * 5
        cands = find_triggers(make_dataset(rows))
        self.assertEqual([c.token for c in cands], [TRIGGER_TOKEN])
        self.assertEqual(cands[0].dominant_label, "positive")

    def test_legit_class_words_not_triggers(self):
        rows = [("excellent keyboard works well", "positive")] * 10
        rows += [("terrible keyboard stopped working", "negative")] * 10
        self.assertEqual(find_triggers(make_dataset(rows)), [])

    def test_cross_label_duplicates_flagged(self):
        rows = [("a perfectly ordinary review text", "positive"),
                ("a perfectly ordinary review text", "negative"),
                ("another distinct review entirely", "positive")]
        pairs, capped = find_duplicates(make_dataset(rows))
        self.assertFalse(capped)
        cross = [p for p in pairs if p.cross_label]
        self.assertEqual(len(cross), 1)

    def test_same_label_duplicates_not_cross_label(self):
        rows = [("same text twice here", "positive"),
                ("same text twice here", "positive")]
        pairs, _ = find_duplicates(make_dataset(rows))
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].cross_label)

    def test_flip_detected_by_neighbours(self):
        rows = [("excellent sturdy keyboard", "positive")] * 8
        rows += [("terrible broken mouse", "negative")] * 8
        rows += [("excellent sturdy headset", "negative")]  # flipped
        disagreements, _ = find_label_disagreements(make_dataset(rows))
        self.assertEqual([d.index for d in disagreements], [17])


class TestEvaluate(unittest.TestCase):
    def test_union_recall_is_total_on_generated_data(self):
        report = evaluate_generated()
        self.assertEqual(report["union"]["false_negatives"], 0)
        self.assertGreaterEqual(report["union"]["precision"], 0.7)
        self.assertEqual(report["triggers"]["false_positives"], 0)


class TestSplitCheck(unittest.TestCase):
    def test_identical_splits_no_shift(self):
        gen = generate(flips=0, triggers=0, duplicates=0)
        ds = Dataset(records=[Record(i + 1, r["text"], r["label"])
                              for i, r in enumerate(gen.rows)])
        report = compare_splits(ds, ds)
        self.assertIn("no meaningful shift", report.verdict_hint)

    def test_forced_shift_detected(self):
        a = make_dataset([("good thing", "positive")] * 40 + [("bad thing", "negative")] * 10)
        b = make_dataset([("good thing", "positive")] * 10 + [("bad thing", "negative")] * 40)
        report = compare_splits(a, b)
        self.assertNotIn("no meaningful shift", report.verdict_hint)


class TestCli(unittest.TestCase):
    def run_cli(self, *argv):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
        return subprocess.run([sys.executable, "-m", "poisonscan.cli", *argv],
                              capture_output=True, text=True, env=env)

    def test_audit_missing_file_exit_1(self):
        self.assertEqual(self.run_cli("audit", "/nonexistent.jsonl").returncode, 1)

    def test_generate_and_audit_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "data.jsonl")
            proc = self.run_cli("generate", "--out", out)
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(os.path.exists(out + ".truth.json"))
            proc = self.run_cli("audit", out)
            self.assertEqual(proc.returncode, 2)
            self.assertIn(TRIGGER_TOKEN, proc.stdout)

    def test_audit_clean_exit_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.jsonl")
            proc = self.run_cli("generate", "--out", out, "--flips", "0",
                                "--triggers", "0", "--duplicates", "0")
            self.assertEqual(proc.returncode, 0)
            proc = self.run_cli("audit", out)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("no poisoning indicators", proc.stdout)

    def test_evaluate_runs(self):
        proc = self.run_cli("evaluate")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("union", proc.stdout)

    def test_compare_split_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "d.jsonl")
            self.run_cli("generate", "--out", out)
            proc = self.run_cli("compare-split", out, out)
            self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()

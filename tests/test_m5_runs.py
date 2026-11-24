"""M5 run-state rules: budgets, verified checkpoints, selection, manifests. No torch needed."""
import json
import tempfile
import unittest
from pathlib import Path

from python_fixer.training import runs


class BudgetTests(unittest.TestCase):
    def test_planned_arithmetic_matches_spec09(self):
        g = json.loads((Path(__file__).resolve().parents[1] / "configs/grpo.json").read_text())["trainer"]
        self.assertEqual(runs.planned_candidates(g["max_steps"], g["generation_batch_size"],
                                                 g["gradient_accumulation_steps"],
                                                 g["per_device_train_batch_size"]), 8000)

    def test_budget_is_hard(self):
        b = runs.CandidateBudget(32)
        b.consume(16); b.consume(16)
        with self.assertRaises(runs.BudgetExceeded):
            b.consume(1)
        self.assertEqual(b.state(), {"max_candidates": 32, "used": 32})
        with self.assertRaises(ValueError):
            runs.CandidateBudget(10, used=11)

    def test_reward_callback_consumes_budget_before_scoring(self):
        try:
            from python_fixer.training.grpo import BehaviorReward
        except ImportError:
            self.skipTest("evaluation extraction unavailable")
        calls = []
        r = BehaviorReward(lambda p, s: (calls.append(p) or 0.0, {}), 2, runs.CandidateBudget(2))
        r(["p", "p"], ["x = 1\n", "x = 2\n"], ["a", "a"])
        with self.assertRaises(runs.BudgetExceeded):
            r(["p", "p"], ["x = 1\n", "x = 2\n"], ["a", "a"])
        self.assertEqual(len(calls), 2, "no scoring after the budget is exhausted")


class SelectionTests(unittest.TestCase):
    def ev(self, step, p, split="validation"):
        return {"step": step, "pass_at_1": p, "split": split}

    def test_best_then_earliest(self):
        got = runs.select_checkpoint([self.ev(50, .2), self.ev(100, .4), self.ev(150, .4), self.ev(200, .3)])
        self.assertEqual(got["step"], 100)

    def test_refuses_heldout_duplicates_and_bad_values(self):
        with self.assertRaises(ValueError):
            runs.select_checkpoint([self.ev(50, .5, "test")])
        with self.assertRaises(ValueError):
            runs.select_checkpoint([self.ev(50, .5), self.ev(50, .6)])
        with self.assertRaises(ValueError):
            runs.select_checkpoint([self.ev(50, float("nan"))])
        with self.assertRaises(ValueError):
            runs.select_checkpoint([])


class CheckpointTests(unittest.TestCase):
    def test_latest_verified_skips_partial_and_tampered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for step in (50, 100, 150):
                c = root / f"checkpoint-{step}"; c.mkdir()
                (c / "adapter.bin").write_bytes(b"w%d" % step)
            runs.seal_checkpoint(root / "checkpoint-50", 50)
            runs.seal_checkpoint(root / "checkpoint-100", 100)
            # 150 is partial (never sealed)
            self.assertEqual(runs.latest_verified_checkpoint(root).name, "checkpoint-100")
            (root / "checkpoint-100" / "adapter.bin").write_bytes(b"corrupt")
            self.assertEqual(runs.latest_verified_checkpoint(root).name, "checkpoint-50")
            (root / "checkpoint-50" / "extra").write_bytes(b"x")
            self.assertIsNone(runs.latest_verified_checkpoint(root))

    def test_empty_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                runs.seal_checkpoint(Path(d), 1)


class ManifestTests(unittest.TestCase):
    def full(self, **kw):
        m = {f: "x" for f in runs.PRE_RUN_FIELDS}
        m.update(arm="E3", seed=17)
        m.update(kw)
        return m

    def test_pre_and_post_run_fields(self):
        self.assertEqual(runs.validate_manifest(self.full()), [])
        self.assertIn("arm", runs.validate_manifest(self.full(arm="E9")))
        self.assertTrue(runs.validate_manifest(self.full(arm="E2", parent_adapter_sha256="")))
        done = self.full(status="COMPLETED")
        problems = runs.validate_manifest(done, completed=True)
        self.assertIn("status COMPLETED with missing fields", problems)
        done.update({f: None for f in runs.POST_RUN_FIELDS if f != "status"})
        self.assertEqual(runs.validate_manifest(done, completed=True), [])


if __name__ == "__main__":
    unittest.main()

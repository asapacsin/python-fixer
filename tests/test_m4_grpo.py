"""M4 CPU tests for the GRPO adapter (spec 04). Scorers here are test doubles; the
real-sandbox integration is scripts/m4_cpu_smoke.py and is recorded separately."""
import json
import math
import tempfile
import unittest

from training_fixtures import ROOT, training_available, train_rows, tiny_tokenizer, tiny_model, cpu_config

GRPO = json.loads((ROOT / "configs/grpo.json").read_text())
SFT = json.loads((ROOT / "configs/sft.json").read_text())
REF = "def absolute(n):\n    return -n if n < 0 else n\n"


@unittest.skipUnless(training_available(), "training stack not installed")
class RewardCallbackTests(unittest.TestCase):
    def setUp(self):
        from python_fixer.training.grpo import BehaviorReward
        self.calls = []
        def scorer(pid, src):
            self.calls.append((pid, src))
            return (1.0 if src == REF else 0.0), {}
        self.reward = BehaviorReward(scorer, 2)

    def test_order_alignment_and_invalid_output(self):
        out = self.reward(["p"] * 4, [REF, "```python\nx\n```", REF + "<|im_end|>", "def f(:\n"],
                          ["a", "a", "b", "b"])
        self.assertEqual(out[:3], [1.0, -0.5, 1.0])
        self.assertEqual([c[0] for c in self.calls], ["a", "b", "b"])
        self.assertEqual(self.calls[1][1], REF, "trailing turn token stripped by strict extraction")

    def test_misaligned_metadata_and_mixed_groups_abort(self):
        with self.assertRaises(ValueError):
            self.reward(["p"] * 2, [REF, REF], ["a"])
        with self.assertRaises(ValueError):
            self.reward(["p"] * 2, [REF, REF], ["a", "b"])
        with self.assertRaises(ValueError):
            self.reward(["p"] * 3, [REF] * 3, ["a"] * 3)

    def test_infrastructure_fault_propagates_not_rewarded(self):
        from python_fixer.sandbox.policy import SandboxUnavailable
        from python_fixer.training.grpo import BehaviorReward
        def down(pid, src):
            raise SandboxUnavailable("worker down")
        with self.assertRaises(SandboxUnavailable):
            BehaviorReward(down, 2)(["p", "p"], [REF, REF], ["a", "a"])

    def test_non_finite_reward_aborts(self):
        from python_fixer.training.grpo import BehaviorReward
        with self.assertRaises(ValueError):
            BehaviorReward(lambda p, s: (float("nan"), {}), 2)(["p", "p"], [REF, REF], ["a", "a"])

    def test_degenerate_group_ledger(self):
        self.reward(["p"] * 4, [REF, REF, "x = (", "y = ("], ["a", "a", "b", "b"])
        s = self.reward.summary()
        self.assertEqual(s["groups"], 2)
        self.assertEqual(s["zero_std"], 1.0)
        self.assertEqual(s["all_pass"], 0.5)


@unittest.skipUnless(training_available(), "training stack not installed")
class GRPOTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from python_fixer.training import grpo, sft
        cls.grpo, cls.sft = grpo, sft
        cls.tok = tiny_tokenizer()
        cls.rows = train_rows()
        cls.prompts, cls.filtered = grpo.build_prompts(cls.rows, cls.tok, GRPO["trainer"]["max_prompt_length"])

    def test_prompts_match_sft_rendering_and_train_only(self):
        examples, _ = self.sft.build_examples(self.rows, self.tok, 10**6)
        by_id = {e["problem_id"]: e["prompt"] for e in examples}
        for p in self.prompts:
            self.assertEqual(p["prompt"], by_id[p["problem_id"]])
        with self.assertRaises(ValueError):
            self.grpo.build_prompts([dict(self.rows[0], split="validation")], self.tok, 4096)

    def test_config_guards(self):
        bad = cpu_config(GRPO); bad["trainer"]["loss_type"] = "dapo"
        with self.assertRaises(ValueError):
            self.grpo.grpo_config(bad, "/tmp/unused")
        with self.assertRaises(ValueError):
            self.grpo.grpo_config(cpu_config(GRPO), "/tmp/unused", steps_per_generation=4)
        with self.assertRaises(ValueError):
            self.grpo.grpo_config(cpu_config(GRPO), "/tmp/unused", num_iterations=2)

    def _run(self, scorer, steps=2):
        from peft import LoraConfig
        with tempfile.TemporaryDirectory() as d:
            model = tiny_model(self.tok, seed=5)
            overrides = dict(max_steps=steps, per_device_train_batch_size=4, generation_batch_size=4,
                             num_generations=4, max_completion_length=8, learning_rate=1e-2, warmup_steps=0,
                             save_steps=1000, use_cpu=True, logging_steps=1,
                             # 8-token CPU completions are all truncated; keep them in the loss here only.
                             mask_truncated_completions=False)
            trainer, reward = self.grpo.make_trainer(model, self.tok, self.prompts[:4], cpu_config(GRPO), d,
                                                     scorer, peft_config=LoraConfig(**SFT["lora"]), **overrides)
            self.sft.assert_only_lora_trainable(trainer.model)
            is_lora = lambda n: "lora_" in n
            base = self.sft.parameter_hashes(trainer.model, lambda n: not is_lora(n))
            lora = self.sft.parameter_hashes(trainer.model, is_lora)
            trainer.train()
            self.assertEqual(base, self.sft.parameter_hashes(trainer.model, lambda n: not is_lora(n)))
            losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
            self.assertTrue(losses and all(math.isfinite(x) for x in losses))
            return lora != self.sft.parameter_hashes(trainer.model, is_lora), reward

    def test_mixed_rewards_update_lora(self):
        changed, reward = self._run(lambda pid, src: (float(len(src) % 3) / 2, {}))
        self.assertTrue(changed)
        self.assertTrue(reward.ledger)
        for g in reward.ledger:
            self.assertEqual(len(g["rewards"]), 4)

    def test_equal_rewards_give_no_update_with_beta_zero(self):
        from unittest import mock
        # Make invalid outputs score the same constant so every group is truly equal.
        with mock.patch.object(self.grpo, "INVALID_OUTPUT_REWARD", 0.0):
            changed, reward = self._run(lambda pid, src: (0.0, {}))
        self.assertEqual(reward.summary()["zero_std"], 1.0)
        self.assertFalse(changed, "equal rewards must give zero advantages and no update at beta=0")


if __name__ == "__main__":
    unittest.main()

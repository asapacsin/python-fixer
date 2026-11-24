"""M3 CPU tests for the SFT adapter (spec 03) on a tiny random Qwen3-architecture model."""
import json
import math
import tempfile
import unittest

from training_fixtures import ROOT, training_available, train_rows, tiny_tokenizer, tiny_model, cpu_config

SFT = json.loads((ROOT / "configs/sft.json").read_text())


@unittest.skipUnless(training_available(), "training stack not installed")
class SFTAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from python_fixer.training import sft
        cls.sft = sft
        cls.tok = tiny_tokenizer()
        cls.rows = train_rows()
        cls.examples, cls.quarantined = sft.build_examples(cls.rows, cls.tok, SFT["trainer"]["max_length"])

    def test_examples_are_prompt_prefix_with_single_eos(self):
        self.assertEqual(len(self.examples) + len(self.quarantined), len(self.rows))
        for ex in self.examples:
            self.assertTrue(ex["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
            self.assertEqual(ex["completion"].count("<|im_end|>"), 1)
            self.assertTrue(ex["completion"].endswith("<|im_end|>"))
            full = self.tok(ex["prompt"] + ex["completion"], add_special_tokens=False)["input_ids"]
            prompt = self.tok(ex["prompt"], add_special_tokens=False)["input_ids"]
            self.assertEqual(full[:len(prompt)], prompt, "prompt tokens must be a prefix")

    def test_prompts_exclude_reference_and_tests(self):
        for ex, row in zip(self.examples, self.rows):
            self.assertNotIn(row["reference_code"], ex["prompt"])
            self.assertNotIn(row["provenance"]["source_uri"], ex["prompt"])
        # Changing tests, provenance or reference must not change any prompt.
        from python_fixer.data.models import messages
        row = self.rows[0]
        mutated = dict(row, tests=[], provenance={}, reference_code="SECRET", source_family_id="x")
        self.assertEqual(self.sft.render_prompt(self.tok, messages(row)),
                         self.sft.render_prompt(self.tok, messages(mutated)))

    def test_non_train_rows_rejected(self):
        row = dict(self.rows[0], split="test")
        with self.assertRaises(ValueError):
            self.sft.build_examples([row], self.tok, 2048)

    def test_overlong_rows_quarantined_not_truncated(self):
        ex, q = self.sft.build_examples(self.rows[:3], self.tok, 20)
        self.assertEqual(ex, [])
        self.assertEqual(len(q), 3)

    def test_rejects_config_that_breaks_masking(self):
        bad = cpu_config(SFT); bad["trainer"]["packing"] = True
        with self.assertRaises(ValueError):
            self.sft.sft_config(bad, "/tmp/unused")

    def test_collated_labels_mask_prompt_and_padding(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self.sft.make_trainer(tiny_model(self.tok), self.tok, self.examples[:4], cpu_config(SFT), d,
                                            per_device_train_batch_size=4, max_steps=1)
            batch = next(iter(trainer.get_train_dataloader()))
            self.sft.check_collated_labels(batch, self.tok)
            # prompt tokens are masked: the first trainable position is after the prompt
            prompt_len = len(self.tok(self.examples[0]["prompt"], add_special_tokens=False)["input_ids"])
            self.assertTrue(bool((batch["labels"][0][:prompt_len] == -100).all()))

    def test_training_changes_only_lora_and_resumes(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            model = tiny_model(self.tok)
            trainer = self.sft.make_trainer(model, self.tok, self.examples[:16], cpu_config(SFT), d,
                                            max_steps=6, save_steps=3, learning_rate=5e-3, warmup_ratio=0.0,
                                            lr_scheduler_type="constant")
            info = self.sft.assert_only_lora_trainable(trainer.model)
            self.assertGreater(info["trainable_params"], 0)
            is_lora = lambda n: "lora_" in n
            base_before = self.sft.parameter_hashes(trainer.model, lambda n: not is_lora(n))
            lora_before = self.sft.parameter_hashes(trainer.model, is_lora)
            result = trainer.train()
            self.assertTrue(math.isfinite(result.training_loss))
            losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
            self.assertTrue(all(math.isfinite(x) for x in losses))
            self.assertLess(losses[-1], losses[0], "toy-set loss should fall when overfitting")
            self.assertEqual(base_before, self.sft.parameter_hashes(trainer.model, lambda n: not is_lora(n)))
            self.assertNotEqual(lora_before, self.sft.parameter_hashes(trainer.model, is_lora))
            final = self.sft.parameter_hashes(trainer.model, is_lora)

            # Resume from step-3 checkpoint reproduces the uninterrupted step-6 adapter.
            resumed = self.sft.make_trainer(tiny_model(self.tok), self.tok, self.examples[:16], cpu_config(SFT),
                                            d, max_steps=6, save_steps=3, learning_rate=5e-3, warmup_ratio=0.0,
                                            lr_scheduler_type="constant")
            resumed.train(resume_from_checkpoint=f"{d}/checkpoint-3")
            self.assertEqual(resumed.state.global_step, 6)
            got = self.sft.parameter_hashes(resumed.model, is_lora)
            for name in final:
                a = dict(trainer.model.named_parameters())[name]
                b = dict(resumed.model.named_parameters())[name]
                self.assertTrue(torch.allclose(a, b, atol=1e-5), name)

    def test_adapter_save_reload_reproduces_logits(self):
        import torch
        from peft import PeftModel
        with tempfile.TemporaryDirectory() as d:
            trainer = self.sft.make_trainer(tiny_model(self.tok, seed=3), self.tok, self.examples[:4],
                                            cpu_config(SFT), d, max_steps=2, learning_rate=5e-3)
            trainer.train()
            trainer.model.save_pretrained(f"{d}/adapter")
            ids = self.tok(self.examples[0]["prompt"], return_tensors="pt")["input_ids"]
            trainer.model.eval()
            with torch.no_grad():
                before = trainer.model(input_ids=ids).logits
            reloaded = PeftModel.from_pretrained(tiny_model(self.tok, seed=3), f"{d}/adapter").eval()
            with torch.no_grad():
                after = reloaded(input_ids=ids).logits
            self.assertTrue(torch.allclose(before, after, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

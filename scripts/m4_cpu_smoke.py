"""M4 CPU integration smoke: tiny random model + TRL GRPO + real mTLS sandbox gateway.

Proves wiring only (prompt rendering -> generation -> strict extraction -> trusted
remote reward -> GRPO update). Random weights: says nothing about model quality.
The real M4 gate additionally needs the pinned Qwen3 model on a GPU node.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def main(argv=None):
    from peft import LoraConfig
    from python_fixer.sandbox.client import RewardClient
    from python_fixer.training import grpo, sft
    from training_fixtures import tiny_tokenizer, tiny_model, cpu_config

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--certs", required=True, help="dir with ca.crt trainer.crt trainer.key")
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in (Path(a.dataset) / "train.jsonl").read_text().splitlines()][:4]
    bindings = {r["problem_id"]: {"suite_sha256": r["test_suite_sha256"],
                                  "image_digest": r["verification"]["image_digest"],
                                  "policy_sha256": r["verification"]["policy_sha256"]} for r in rows}
    client = RewardClient(a.endpoint, ca=f"{a.certs}/ca.crt", certificate=f"{a.certs}/trainer.crt",
                          key=f"{a.certs}/trainer.key", bindings=bindings, run_id="m4-cpu-smoke")
    tok = tiny_tokenizer()
    cfg = cpu_config(json.loads((ROOT / "configs/grpo.json").read_text()))
    prompts, _ = grpo.build_prompts(rows, tok, cfg["trainer"]["max_prompt_length"])

    # 1) Known candidates through the same callback and real gateway.
    probe = grpo.BehaviorReward(client.score, 2)
    known = probe(["p"] * 4, [rows[0]["reference_code"], rows[0]["buggy_code"],
                              rows[1]["reference_code"], "```\nnot a module\n```"],
                  [rows[0]["problem_id"]] * 2 + [rows[1]["problem_id"]] * 2)

    # 2) Two GRPO updates: 4 prompts x 4 completions, rewards from the real sandbox.
    lora = LoraConfig(**json.loads((ROOT / "configs/sft.json").read_text())["lora"])
    trainer, reward = grpo.make_trainer(tiny_model(tok, seed=7), tok, prompts, cfg, a.output, client.score,
                                        peft_config=lora, max_steps=2, per_device_train_batch_size=4,
                                        generation_batch_size=16, num_generations=4, max_completion_length=24,
                                        warmup_steps=0, save_steps=1000, use_cpu=True, logging_steps=1)
    is_lora = lambda n: "lora_" in n
    before = sft.parameter_hashes(trainer.model, is_lora)
    base = sft.parameter_hashes(trainer.model, lambda n: not is_lora(n))
    trainer.train()
    report = {"kind": "m4-cpu-grpo-smoke", "known_candidate_rewards": known,
              "known_expected": [1.0, 0.0, 1.0, -0.5],
              "global_step": trainer.state.global_step,
              "group_summary": reward.summary(), "groups": reward.ledger,
              "lora_changed": before != sft.parameter_hashes(trainer.model, is_lora),
              "base_unchanged": base == sft.parameter_hashes(trainer.model, lambda n: not is_lora(n)),
              "log_history": trainer.state.log_history,
              "limitations": "random tiny model; CPU; no GPU/Qwen; not M4 completion evidence"}
    Path(a.report).write_text(json.dumps(report, indent=1, default=str))
    ok = known == report["known_expected"] and report["global_step"] == 2 and report["base_unchanged"]
    print(json.dumps({k: report[k] for k in ("known_candidate_rewards", "global_step", "group_summary",
                                             "lora_changed", "base_unchanged")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

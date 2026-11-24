"""GRPO adapter (spec 04): TRL GRPOTrainer + trusted behavior-v1 reward callback.

The reward callback never executes code. It extracts the candidate text strictly and
asks a trusted scorer (normally `sandbox.client.RewardClient.score`, train-only
bindings over mTLS). Infrastructure faults propagate and stop training: they are
never converted into rewards.
"""
from __future__ import annotations

import math
from typing import Callable

from python_fixer.data.models import model_view
from python_fixer.evaluation.extraction import extract_module
from .sft import render_prompt

INVALID_OUTPUT_REWARD = -0.5  # reward v1 penalty for INVALID_OUTPUT (spec 05)

Scorer = Callable[[str, str], tuple]


def build_prompts(rows, tokenizer, max_prompt_length: int) -> tuple[list[dict], list[dict]]:
    """String prompts identical to SFT/eval rendering; over-long prompts are filtered, never truncated."""
    kept, filtered = [], []
    for row in rows:
        view = model_view(row, "rl")  # train-only, verified
        prompt = render_prompt(tokenizer, view["prompt"])
        n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n > max_prompt_length:
            filtered.append({"problem_id": view["problem_id"], "tokens": n})
        else:
            kept.append({"prompt": prompt, "problem_id": view["problem_id"]})
    return kept, filtered


class BehaviorReward:
    """Callable with TRL's reward signature; records per-group degeneracy statistics."""

    __name__ = "behavior_reward"

    def __init__(self, scorer: Scorer, num_generations: int, budget=None):
        if num_generations < 2:
            raise ValueError("GRPO needs at least two generations per prompt")
        self.scorer = scorer
        self.budget = budget  # runs.CandidateBudget: counts every generated candidate
        self.G = num_generations
        self.ledger: list[dict] = []

    def __call__(self, prompts, completions, problem_id, **kwargs) -> list[float]:
        if not (len(prompts) == len(completions) == len(problem_id)):
            raise ValueError("reward metadata misaligned with completions")
        if len(completions) % self.G:
            raise ValueError("completion count is not a multiple of num_generations")
        if self.budget is not None:
            self.budget.consume(len(completions))  # raises BudgetExceeded before any scoring
        rewards = []
        for pid, completion in zip(problem_id, completions):
            if not isinstance(completion, str):
                raise ValueError("expected string completions (non-conversational prompts)")
            extracted = extract_module(completion)
            if extracted.status != "VALID":
                rewards.append(INVALID_OUTPUT_REWARD)
                continue
            reward = self.scorer(pid, extracted.module.decode("utf-8"))[0]
            if not isinstance(reward, float) or not math.isfinite(reward):
                raise ValueError("scorer returned a non-finite reward")
            rewards.append(reward)
        self._record(problem_id, rewards)
        return rewards

    def _record(self, problem_id, rewards):
        for start in range(0, len(rewards), self.G):
            ids = set(problem_id[start:start + self.G])
            group = rewards[start:start + self.G]
            if len(ids) != 1:
                raise ValueError("a generation group mixes problems: " + ", ".join(sorted(ids)))
            self.ledger.append({"problem_id": ids.pop(), "rewards": group,
                                "zero_std": max(group) == min(group),
                                "all_pass": all(r == 1.0 for r in group),
                                "all_fail": all(r <= 0.0 for r in group)})

    def summary(self) -> dict:
        n = len(self.ledger)
        frac = lambda k: (sum(g[k] for g in self.ledger) / n) if n else 0.0
        return {"groups": n, "zero_std": frac("zero_std"), "all_pass": frac("all_pass"),
                "all_fail": frac("all_fail")}


def grpo_config(cfg: dict, output_dir: str, **overrides):
    from trl import GRPOConfig
    args = dict(cfg["trainer"])
    args.update(overrides)
    if args.get("loss_type") != "dr_grpo" or args.get("remove_unused_columns") is not False:
        raise ValueError("spec 04 requires loss_type=dr_grpo and remove_unused_columns=False")
    if "steps_per_generation" in args and "generation_batch_size" in args:
        raise ValueError("set generation_batch_size or steps_per_generation, not both")
    if args.get("num_iterations", 1) != 1:
        raise ValueError("spec 04 primary arm uses one update per generation (num_iterations=1)")
    if "generation_batch_size" not in args:
        raise ValueError("generation_batch_size is required (spec 04 budget arithmetic)")
    if args["generation_batch_size"] % args["num_generations"]:
        raise ValueError("generation batch must be divisible by num_generations")
    args.setdefault("seed", cfg.get("seed", 17))
    return GRPOConfig(output_dir=output_dir, **args)


def make_trainer(model, tokenizer, prompts: list[dict], cfg: dict, output_dir: str, scorer: Scorer,
                 peft_config=None, budget=None, **overrides):
    """E3: pass a LoraConfig. E2: pass a PeftModel already holding the E1 adapter and peft_config=None."""
    from datasets import Dataset
    from trl import GRPOTrainer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = grpo_config(cfg, output_dir, **overrides)
    reward = BehaviorReward(scorer, args.num_generations, budget)
    is_peft = hasattr(model, "peft_config")
    if is_peft and peft_config is not None:
        raise ValueError("model already has an adapter; do not double-wrap")
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=args, reward_funcs=[reward],
                          train_dataset=Dataset.from_list(prompts), peft_config=peft_config)
    return trainer, reward

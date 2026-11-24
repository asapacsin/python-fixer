"""SFT adapter (spec 03): prompt/completion rows, LoRA config and TRL SFTTrainer.

This module only wires established libraries. It never downloads weights; callers
pass an already-loaded model and tokenizer (the GPU run pins them by commit).
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from python_fixer.data.models import model_view

IGNORE_INDEX = -100


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def build_examples(rows: Iterable[dict], tokenizer, max_length: int) -> tuple[list[dict], list[dict]]:
    """Return (examples, quarantined). Never truncates: over-long rows are excluded."""
    eos = tokenizer.eos_token
    if not eos:
        raise ValueError("tokenizer must define the turn-ending EOS token")
    examples, quarantined = [], []
    for row in rows:
        view = model_view(row, "sft")  # rejects non-train / unverified rows
        prompt = render_prompt(tokenizer, view["prompt"])
        code = view["completion"]
        if not code.strip():
            raise ValueError("empty SFT target: " + row["problem_id"])
        if eos in code:
            raise ValueError("target already contains EOS: " + row["problem_id"])
        completion = code + eos  # exactly one turn-ending token
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        n_total = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
        if n_total > max_length:
            quarantined.append({"problem_id": row["problem_id"], "tokens": n_total, "reason": "exceeds max_length"})
            continue
        if n_total <= n_prompt:
            raise ValueError("completion produced no tokens: " + row["problem_id"])
        examples.append({"problem_id": row["problem_id"], "prompt": prompt, "completion": completion})
    return examples, quarantined


def lora_config(cfg: dict):
    from peft import LoraConfig
    return LoraConfig(**cfg["lora"])


def sft_config(cfg: dict, output_dir: str, **overrides):
    from trl import SFTConfig
    args = dict(cfg["trainer"])
    if not (args.get("completion_only_loss") is True and args.get("assistant_only_loss") is False
            and args.get("packing") is False):
        raise ValueError("spec 03 requires completion_only_loss, no assistant mask, no packing")
    args.update(overrides)
    args.setdefault("seed", cfg.get("seed", 17))
    args.setdefault("dataset_kwargs", {"skip_prepare_dataset": False})
    return SFTConfig(output_dir=output_dir, **args)


def make_trainer(model, tokenizer, train_examples: list[dict], cfg: dict, output_dir: str,
                 eval_examples: list[dict] | None = None, **overrides):
    from datasets import Dataset
    from trl import SFTTrainer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    strip = lambda xs: Dataset.from_list([{"prompt": x["prompt"], "completion": x["completion"]} for x in xs])
    trainer = SFTTrainer(model=model, processing_class=tokenizer, peft_config=lora_config(cfg),
                         args=sft_config(cfg, output_dir, **overrides), train_dataset=strip(train_examples),
                         eval_dataset=strip(eval_examples) if eval_examples else None)
    assert_only_lora_trainable(trainer.model)
    return trainer


def assert_only_lora_trainable(model) -> dict:
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    bad = [n for n in trainable if "lora_" not in n]
    if bad:
        raise ValueError("non-LoRA parameters require grad: " + ", ".join(bad[:5]))
    return {"trainable_tensors": len(trainable),
            "trainable_params": sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)}


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()


def parameter_hashes(model, predicate) -> dict[str, str]:
    return {n: tensor_sha256(p) for n, p in model.named_parameters() if predicate(n)}


def check_collated_labels(batch, tokenizer) -> None:
    """Assert spec-03 masking on an actual collated batch."""
    ids, labels, attn = batch["input_ids"], batch["labels"], batch["attention_mask"]
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    for i in range(ids.shape[0]):
        trainable = labels[i] != IGNORE_INDEX
        if int(trainable.sum()) == 0:
            raise ValueError("example has no trainable target tokens")
        if bool((trainable & (attn[i] == 0)).any()):
            raise ValueError("padding tokens are trainable")
        target = ids[i][trainable].tolist()
        if target.count(eos_id) != 1 or target[-1] != eos_id:
            raise ValueError("target must end with exactly one EOS")
        first = int(trainable.nonzero()[0])
        if bool(trainable[first:int(trainable.nonzero()[-1]) + 1].logical_not().any()):
            raise ValueError("target tokens are not contiguous")

"""Generate paired greedy candidates for the exploratory validation comparison.

This script performs generation only. Candidate execution belongs in the isolated
CPU worker; it must never be run on the GPU host or Windows host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from python_fixer.data.models import messages


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise SystemExit("input must contain validation rows only")
    if len({row["problem_id"] for row in rows}) != len(rows):
        raise SystemExit("duplicate validation problem_id")
    return rows


def generate_arm(rows: list[dict], tokenizer, model, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generation = model.generation_config
    eos_token_id = generation.eos_token_id
    pad_token_id = generation.pad_token_id
    with output.open("x", encoding="utf-8") as stream, torch.inference_mode():
        # One excluded warm-up per arm, matching the protocol's timing convention.
        warm_prompt = tokenizer.apply_chat_template(
            messages(rows[0]), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        warm_inputs = tokenizer(warm_prompt, return_tensors="pt").to(model.device)
        model.generate(**warm_inputs, do_sample=False, max_new_tokens=512,
                       eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        torch.cuda.synchronize()

        for row in rows:
            prompt = tokenizer.apply_chat_template(
                messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_tokens = inputs.input_ids.shape[1]
            torch.cuda.synchronize()
            began = time.perf_counter()
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=512,
                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            )
            torch.cuda.synchronize()
            elapsed_ms = round((time.perf_counter() - began) * 1000, 2)
            token_ids = generated[0, input_tokens:]
            completion = tokenizer.decode(token_ids, skip_special_tokens=False).replace("<|endoftext|>", "")
            raw = completion.encode("utf-8")
            record = {
                "problem_id": row["problem_id"],
                "candidate_sha256": hashlib.sha256(raw).hexdigest(),
                "generated_tokens": len(token_ids),
                "generation_ms": elapsed_ms,
                "finish_reason": "eos" if len(token_ids) and int(token_ids[-1]) in (
                    [eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or [])
                ) else "length_or_other",
                "completion": completion,
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            print(row["problem_id"], record["generated_tokens"], record["generation_ms"], flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="pinned base model directory")
    parser.add_argument("--adapter", required=True, help="completed SFT LoRA adapter directory")
    parser.add_argument("--data", required=True, help="validation.jsonl only")
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--sft-output", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        raise SystemExit("pinned tokenizer has no chat template")

    print("loading baseline", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    generate_arm(rows, tokenizer, base, Path(args.base_output))
    del base
    torch.cuda.empty_cache()

    print("loading SFT adapter", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    sft = PeftModel.from_pretrained(base, args.adapter).eval()
    generate_arm(rows, tokenizer, sft, Path(args.sft_output))
    print("generation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E0 baseline on the VALIDATION split only (test stays sealed), CPU, untuned Qwen3-0.6B.

Greedy decode (enable_thinking=False) plus G sampled candidates per problem with the
frozen GRPO sampling settings, each scored by the real gVisor sandbox. Output is a
new JSON artifact; it informs dataset sizing (power) and GRPO signal, never selection.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from python_fixer.data.models import messages
from python_fixer.evaluation.extraction import extract_module
from python_fixer.sandbox.docker import DockerSandbox
from python_fixer.sandbox.gateway import execute_suite
from python_fixer.sandbox.policy import Policy

ROOT = Path(__file__).resolve().parents[1]


def score(sandbox, row, completion):
    ext = extract_module(completion)
    if ext.status != "VALID":
        return {"status": "INVALID_OUTPUT", "reason": ext.reason, "solved": False}
    out = execute_suite(sandbox, row, ext.module.decode("utf-8"))
    solved = out["status"] == "OK" and len(out["passed_ids"]) == len(row["tests"])
    return {"status": out["status"], "solved": solved, "passed": len(out.get("passed_ids", [])),
            "tests": len(row["tests"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "verification/evidence/m1-v9/m1-fixtures-v9/validation.jsonl"))
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if "test" in Path(a.data).name:
        raise SystemExit("test split is sealed")
    out = Path(a.output)
    if out.exists():
        raise SystemExit("output exists")
    cfg = json.loads((ROOT / "configs/grpo.json").read_text())["trainer"]
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l]
    assert all(r["split"] == "validation" for r in rows)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    sandbox = DockerSandbox(Policy.load(ROOT / "configs/sandbox.json"))
    results, t0 = [], time.time()
    for i, row in enumerate(rows):
        prompt = tok.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        ids = tok(prompt, return_tensors="pt")
        n = ids.input_ids.shape[1]
        with torch.no_grad():
            g = model.generate(**ids, do_sample=False, max_new_tokens=cfg["max_completion_length"])
            s = model.generate(**ids, do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"],
                               top_k=cfg["top_k"], num_return_sequences=a.samples,
                               max_new_tokens=cfg["max_completion_length"])
        dec = lambda t: tok.decode(t[n:], skip_special_tokens=False).replace("<|endoftext|>", "")
        greedy = score(sandbox, row, dec(g[0]))
        greedy["text_head"] = dec(g[0])[:120]
        sampled = [score(sandbox, row, dec(x)) for x in s]
        results.append({"problem_id": row["problem_id"], "greedy": greedy, "sampled": sampled})
        print(i, row["problem_id"], greedy["status"], greedy["solved"],
              sum(x["solved"] for x in sampled), f"{time.time()-t0:.0f}s", flush=True)
    N = len(results)
    k = sum(r["greedy"]["solved"] for r in results)
    mixed = sum(0 < sum(x["solved"] for x in r["sampled"]) < a.samples for r in results)
    report = {
        "evidence_kind": "OBSERVED_CPU_E0_VALIDATION", "split": "validation", "problems": N,
        "model": a.model, "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "greedy_solved": k, "greedy_pass_rate": k / N,
        "sampled_pass_rate": sum(x["solved"] for r in results for x in r["sampled"]) / (N * a.samples),
        "problems_with_mixed_group_outcomes": mixed, "group_size": a.samples,
        "wall_seconds": round(time.time() - t0), "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

"""M6 data-access and contamination audits (spec 06). Text-only; nothing is executed."""
from __future__ import annotations

import json
import re

IDENTITY_KEYS = ("problem_id", "source_family_id", "source_function_id")


def data_access_audit(training_rows: list[dict], heldout_rows: list[dict],
                      model_inputs: list[str]) -> list[str]:
    """Held-out identities and reference code must not reach any training artifact."""
    issues = []
    for key in IDENTITY_KEYS:
        overlap = {r[key] for r in training_rows} & {r[key] for r in heldout_rows}
        if overlap:
            issues.append(f"{key} shared with held-out: {sorted(overlap)[:3]}")
    blob = "\n".join(model_inputs)
    for r in heldout_rows:
        if r["reference_code"].strip() in blob:
            issues.append("held-out reference code in training input: " + r["problem_id"])
        for t in r.get("tests", []):
            needle = json.dumps(t.get("args"), sort_keys=True)
            if len(needle) >= 24 and needle in blob:  # short literals collide by chance
                issues.append("held-out test arguments in training input: " + r["problem_id"])
                break
    return issues


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", text)


def ngram_overlap(target: str, corpus: list[str], n: int = 13) -> float:
    """Fraction of the target's token n-grams found anywhere in the corpus."""
    toks = _tokens(target)
    grams = {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    if not grams:
        return 0.0
    seen = set()
    for doc in corpus:
        dt = _tokens(doc)
        seen.update(tuple(dt[i:i + n]) for i in range(len(dt) - n + 1))
    return len(grams & seen) / len(grams)


def contamination_report(heldout_rows: list[dict], corpus: list[str], n: int = 13,
                         threshold: float = 0.5) -> dict:
    per = {r["problem_id"]: ngram_overlap(r["reference_code"], corpus, n) for r in heldout_rows}
    flagged = sorted(p for p, v in per.items() if v >= threshold)
    return {"n": n, "threshold": threshold, "overlap": per, "flagged": flagged,
            "note": "flags exact textual overlap only; pretraining contamination is not measurable here"}

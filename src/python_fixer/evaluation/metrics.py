"""Repair metrics with explicit denominators and no inferred missing outcomes."""
from __future__ import annotations
from collections import defaultdict
import math
from statistics import mean


def pass_at_k(n: int, c: int, k: int) -> float:
    if any(type(v) is not int for v in (n,c,k)) or not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("pass@k requires integer n>=k>=1 and 0<=c<=n")
    if n-c < k:
        return 1.0
    return 1.0-math.prod((n-c-i)/(n-i) for i in range(k))


def quantile(values: list[float], p: float) -> float:
    if not values or not 0 <= p <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("finite nonempty values and 0<=p<=1 required")
    ordered=sorted(values)
    x=(len(ordered)-1)*p
    lo=math.floor(x)
    hi=math.ceil(x)
    return ordered[lo]+(x-lo)*(ordered[hi]-ordered[lo])


def summarize(records: list[dict], problems: dict[str,dict], mode: str, k_values: list[int]) -> dict:
    if not records:
        raise ValueError("zero records cannot validate an experiment")
    groups=defaultdict(list)
    for row in records:
        groups[(row["problem_id"],row["sampling_seed"])].append(row)
    per_problem={}
    for (problem,seed),group in groups.items():
        successes=sum(r["all_tests_pass"] for r in group)
        for k in k_values:
            per_problem.setdefault(problem,defaultdict(list))[k].append(pass_at_k(len(group),successes,k))
    # Sampling repeats are averaged within a problem, never treated as extra problems.
    metrics={f"pass@{k}":mean(mean(v[k]) for v in per_problem.values()) for k in k_values}
    by_family=defaultdict(list)
    for problem,values in per_problem.items():
        by_family[problems[problem]["source_family_id"]].append(mean(values[1]))
    counts=len(records)
    complete=[r for r in records if r["regressed_passed_count"] is not None]
    complete_tests=sum(problems[r["problem_id"]]["baseline_passed_count"] for r in complete)
    all_tests=sum(problems[r["problem_id"]]["baseline_passed_count"] for r in records)
    result={"mode":mode,"candidate_count":counts,"problem_count":len(per_problem),
        "family_count":len(by_family),"metrics":metrics,
        "family_weighted_pass@1":mean(mean(v) for v in by_family.values()),
        "compilation_rate":sum(r["compiled"] for r in records)/counts,
        "observed_regression_rate":sum(r["regression"] for r in records)/counts,
        "conservative_regression_rate":sum(r["regression"] or r["status"]!="OK" for r in records)/counts,
        "complete_execution_rate":sum(r["status"]=="OK" for r in records)/counts,
        "average_generation_tokens":mean(r["generated_tokens"] for r in records),
        "generation_latency_ms":{"median":quantile([r["generation_ms"] for r in records],.5),
                                  "p95":quantile([r["generation_ms"] for r in records],.95)},
        "failure_rates":{status:sum(r["status"]==status for r in records)/counts
            for status in ("INVALID_OUTPUT","COMPILE_ERROR","POLICY_VIOLATION","TIMEOUT","RESOURCE_LIMIT")},
        "per_test_regression_rate":sum(r["regressed_passed_count"] for r in complete)/complete_tests if complete_tests else None,
        "per_test_regression_observed_denominator":complete_tests,
        "per_test_regression_observed_numerator":sum(r["regressed_passed_count"] for r in complete),
        "conservative_per_test_regression_rate":sum(r["regressed_passed_count"] if r["regressed_passed_count"] is not None
            else problems[r["problem_id"]]["baseline_passed_count"] for r in records)/all_tests,
        "per_test_regression_note":"observed denominator includes only complete executions; conservative rate treats unexecuted tests as failures",
        "compute_cost":None,"compute_cost_note":"requires separately audited cost ledger; not inferred from latency"}
    for label in ("bug_category","difficulty"):
        buckets=defaultdict(list)
        for problem,values in per_problem.items():
            buckets[problems[problem][label]].append(mean(values[1]))
        result["by_"+label]={name:{"problem_count":len(values),"pass@1":mean(values)} for name,values in sorted(buckets.items())}
    return result

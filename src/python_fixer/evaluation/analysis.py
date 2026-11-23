"""Analyze a complete frozen grid, without generation or test execution."""
from __future__ import annotations
from collections import defaultdict
from statistics import mean
from python_fixer.common import digest
from .metrics import summarize, quantile
from .statistics import paired_cluster_analysis, holm_adjust


def analyze_validated(plan: dict, records: list[dict]) -> dict:
    problems={p["problem_id"]:p for p in plan["problems"]}
    by_arm=defaultdict(lambda:defaultdict(list))
    for row in records:
        by_arm[row["arm"]][row["train_seed"]].append(row)
    summaries={}
    outcomes={}
    for arm,seeds in sorted(by_arm.items()):
        per_seed={}
        outcomes[arm]={}
        for seed,rows in seeds.items():
            label="base" if seed is None else str(seed)
            per_seed[label]=summarize(rows,problems,plan["generation"]["mode"],plan["generation"]["k"])
            rates=defaultdict(list)
            for row in rows:
                rates[row["problem_id"]].append(int(row["all_tests_pass"]))
            outcomes[arm][seed]={problem:mean(values) for problem,values in rates.items()}
        summaries[arm]={"seed_count":len(per_seed),"per_seed":per_seed,
            "metrics":{f"pass@{k}":mean(v["metrics"][f"pass@{k}"] for v in per_seed.values()) for k in plan["generation"]["k"]},
            "candidate_count":sum(v["candidate_count"] for v in per_seed.values()),
            "problem_count":len(problems),"family_count":len({p["source_family_id"] for p in problems.values()}),
            "observed_regression_rate":mean(v["observed_regression_rate"] for v in per_seed.values()),
            "conservative_regression_rate":mean(v["conservative_regression_rate"] for v in per_seed.values())}
        summary=summaries[arm]
        for metric in ("compilation_rate","family_weighted_pass@1","average_generation_tokens","conservative_per_test_regression_rate"):
            summary[metric]=mean(v[metric] for v in per_seed.values())
        denominator=sum(v["per_test_regression_observed_denominator"] for v in per_seed.values())
        numerator=sum(v["per_test_regression_observed_numerator"] for v in per_seed.values())
        summary.update(per_test_regression_rate=numerator/denominator if denominator else None,
                       per_test_regression_observed_denominator=denominator)
        summary["failure_rates"]={status:mean(v["failure_rates"][status] for v in per_seed.values())
                                  for status in next(iter(per_seed.values()))["failure_rates"]}
        times=[r["generation_ms"] for rows in seeds.values() for r in rows]
        summary["generation_latency_ms"]={"median":quantile(times,.5),"p95":quantile(times,.95)}
        for label in ("bug_category","difficulty"):
            first=next(iter(per_seed.values()))["by_"+label]
            summary["by_"+label]={name:{"problem_count":value["problem_count"],
                "pass@1":mean(v["by_"+label][name]["pass@1"] for v in per_seed.values())} for name,value in first.items()}
    comparisons={}
    for comparison in plan["comparisons"]:
        left,right=comparison["left"],comparison["right"]
        lseeds,rseeds=outcomes[left],outcomes[right]
        paired_seeds=list(rseeds) if set(lseeds)=={None} else list(lseeds)
        differences=defaultdict(lambda:defaultdict(list))
        for problem in sorted(problems):
            family=problems[problem]["source_family_id"]
            for seed in sorted(paired_seeds,key=lambda v:-1 if v is None else v):
                lv=lseeds[None if set(lseeds)=={None} else seed][problem]
                rv=rseeds[None if set(rseeds)=={None} else seed][problem]
                differences[family]["base" if seed is None else str(seed)].append(lv-rv)
        result=paired_cluster_analysis(differences,**plan["bootstrap"])
        result.update(left=left,right=right,primary=comparison["primary"],
            regression_increase=summaries[left]["conservative_regression_rate"]-summaries[right]["conservative_regression_rate"])
        comparisons[comparison["id"]]=result
    primary={key:c["permutation_p"] for key,c in comparisons.items() if c["primary"]}
    adjusted=holm_adjust(primary) if primary else {}
    eligible=(plan["study_scope"]=="confirmatory" and plan["evidence_kind"]=="observed-model" and plan["generation"]["mode"]=="primary-greedy")
    for key,result in comparisons.items():
        result["holm_adjusted_p"]=adjusted.get(key)
        numeric=(result["difference"]>=plan["meaningful_gain_absolute"] and
                 result["simultaneous_ci975"][0]>0 and
                 result["regression_increase"]<=plan["max_regression_increase_absolute"] and
                 adjusted.get(key,1.0)<plan["holm_family_alpha"])
        result["preregistered_numeric_criteria_met"]=bool(numeric and eligible and result["primary"])
        result["confirmatory_criteria_applicable"]=eligible and result["primary"]
        result["claim_note"]="Numeric criterion only; independent execution and scientific review still required."
    return {"schema_version":1,"status":"ANALYZED_COMPLETE_GRID","study_id":plan["study_id"],
        "plan_sha256":digest(plan),"evidence_kind":plan["evidence_kind"],"study_scope":plan["study_scope"],
        "mode":plan["generation"]["mode"],"candidate_count":len(records),"arms":summaries,
        "comparisons":comparisons,"independent_run_authentication":"NOT_VERIFIED_BY_ANALYSIS",
        "notes":["No model generation, training or candidate execution performed by analysis.",
                 "Sampling repeats are averaged within problems; trained checkpoints are not pooled for pass@k.",
                 "Incomplete execution contributes to conservative regression rate, never a false regression observation.",
                 "Hash integrity does not authenticate an actual model or hardware run."]}

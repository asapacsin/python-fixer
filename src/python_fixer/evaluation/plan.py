from __future__ import annotations
from itertools import product
from jsonschema import Draft202012Validator
from python_fixer.common import resource_path, strict_loads, digest, canonical_bytes


def validate_plan(plan: dict) -> None:
    canonical_bytes(plan)  # Reject non-finite values from direct Python callers too.
    schema=strict_loads(resource_path("schemas/evaluation-plan.schema.json").read_text())
    Draft202012Validator(schema).validate(plan)
    integer_values=[plan["schema_version"],plan["generation"]["n"],plan["generation"]["max_new_tokens"],
                    plan["inference_profile"]["batch_size"],plan["bootstrap"]["replicates"],plan["bootstrap"]["seed"],
                    *plan["generation"]["k"]]
    integer_values += [s for s in plan["generation"]["sampling_seeds"] if s is not None]
    integer_values += [c["train_seed"] for a in plan["arms"] for c in a["checkpoints"] if c["train_seed"] is not None]
    integer_values += [p[k] for p in plan["problems"] for k in ("baseline_failed_count","baseline_passed_count")]
    if "top_k" in plan["generation"]:
        integer_values.append(plan["generation"]["top_k"])
    if any(type(value) is not int for value in integer_values):
        raise ValueError("integer protocol fields cannot use booleans or floating-point representations")
    if plan["status"]!="FROZEN":
        raise ValueError("analysis requires an operator-frozen plan")
    arms={a["arm"]:a for a in plan["arms"]}
    if len(arms)!=len(plan["arms"]):
        raise ValueError("duplicate arm")
    by_arm={}
    for arm,entry in arms.items():
        seeds={c["train_seed"]:c for c in entry["checkpoints"]}
        if len(seeds)!=len(entry["checkpoints"]):
            raise ValueError("duplicate checkpoint seed")
        if (arm in {"E0","E0-P"} and set(seeds)!={None}) or (arm not in {"E0","E0-P"} and None in seeds):
            raise ValueError("base controls have null seed; trained arms require real seed IDs")
        for checkpoint in seeds.values():
            if arm in {"E0","E0-P"} and checkpoint["checkpoint_sha256"]!=plan["base_checkpoint_sha256"]:
                raise ValueError("base controls must use the exact base checkpoint")
            if arm in {"E1","E3"} and checkpoint.get("parent_checkpoint_sha256")!=plan["base_checkpoint_sha256"]:
                raise ValueError("E1 and E3 require the exact base initializer parent")
        by_arm[arm]=seeds
    if "E0" in arms and "E0-P" in arms:
        if by_arm["E0"][None]["checkpoint_sha256"]!=by_arm["E0-P"][None]["checkpoint_sha256"]:
            raise ValueError("stronger-prompt baseline must use the exact base checkpoint")
    if "E2" in arms:
        if "E1" not in arms or set(by_arm["E2"])!=set(by_arm["E1"]):
            raise ValueError("E2 requires matching E1 seed checkpoints")
        for seed,checkpoint in by_arm["E2"].items():
            if checkpoint.get("parent_checkpoint_sha256")!=by_arm["E1"][seed]["checkpoint_sha256"]:
                raise ValueError("E2 parent does not match selected E1 checkpoint")
    problems={p["problem_id"]:p for p in plan["problems"]}
    if len(problems)!=len(plan["problems"]):
        raise ValueError("duplicate problem ID")
    for problem in problems.values():
        if set(problem["prompt_sha256_by_arm"])!=set(arms):
            raise ValueError("each problem needs a frozen prompt digest for every arm")
        prompts=problem["prompt_sha256_by_arm"]
        original={value for arm,value in prompts.items() if arm!="E0-P"}
        if len(original)!=1 or ("E0-P" in prompts and prompts["E0-P"] in original):
            raise ValueError("standard arms share one prompt; stronger baseline requires a distinct fixed prompt")
    comparisons={c["id"]:c for c in plan["comparisons"]}
    if len(comparisons)!=len(plan["comparisons"]):
        raise ValueError("duplicate comparison ID")
    for comparison in comparisons.values():
        left,right=comparison["left"],comparison["right"]
        if left==right or left not in arms or right not in arms:
            raise ValueError("comparison references absent or identical arms")
        if None not in by_arm[left] and None not in by_arm[right] and set(by_arm[left])!=set(by_arm[right]):
            raise ValueError("trained contrasts require matched seed sets")
    generation=plan["generation"]
    if 1 not in generation["k"] or max(generation["k"])>generation["n"]:
        raise ValueError("pass@1 required and k cannot exceed n")
    if generation["mode"]=="primary-greedy":
        if generation["do_sample"] or generation["n"]!=1 or generation["sampling_seeds"]!=[None] or generation["k"]!=[1]:
            raise ValueError("primary greedy is exactly one unsampled completion per problem")
    else:
        if not generation["do_sample"] or None in generation["sampling_seeds"] or not all(k in generation for k in ("temperature","top_p","top_k")):
            raise ValueError("sampled evaluation requires explicit settings and seeds")
    if plan["study_scope"]=="confirmatory":
        if plan["evidence_kind"]!="observed-model" or set(arms)!={"E0","E0-P","E1","E2","E3"}:
            raise ValueError("confirmatory study requires all five observed arms")
        if any(set(by_arm[a])!={17,29,43} for a in ("E1","E2","E3")):
            raise ValueError("confirmatory study requires three preregistered seeds")
        primary={(c["left"],c["right"]) for c in comparisons.values() if c["primary"]}
        if primary!={("E1","E0"),("E2","E1")} or sum(c["primary"] for c in comparisons.values())!=2:
            raise ValueError("confirmatory primary comparison family changed")
        if (plan["meaningful_gain_absolute"],plan["max_regression_increase_absolute"],plan["holm_family_alpha"])!=(.05,.02,.05):
            raise ValueError("confirmatory thresholds changed")
        if generation["mode"]=="secondary-sampled" and (generation["n"]!=20 or generation["k"]!=[1,5,10] or generation["sampling_seeds"]!=[101,202,303] or generation["temperature"]!=.7 or generation["top_p"]!=.8 or generation["top_k"]!=20):
            raise ValueError("confirmatory secondary generation settings changed")
        if plan["bootstrap"]["replicates"]!=10000 or plan["bootstrap"]["seed"]!=987:
            raise ValueError("confirmatory bootstrap settings changed")
    if plan["evidence_kind"]=="synthetic-test" and plan["study_scope"]!="synthetic-test":
        raise ValueError("synthetic evidence cannot masquerade as a research study")
    expected=sum(len(a["checkpoints"]) for a in arms.values())*len(problems)*len(generation["sampling_seeds"])*generation["n"]
    if expected>1000000:
        raise ValueError("evaluation grid exceeds bounded one-million-record analysis capacity")


def expected_keys(plan: dict) -> set[tuple]:
    return {(arm["arm"],checkpoint["train_seed"],problem["problem_id"],seed,index)
        for arm in plan["arms"] for checkpoint in arm["checkpoints"] for problem in plan["problems"]
        for seed in plan["generation"]["sampling_seeds"] for index in range(plan["generation"]["n"])}

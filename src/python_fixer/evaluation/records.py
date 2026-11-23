from __future__ import annotations
import hashlib
from pathlib import Path
from jsonschema import Draft202012Validator
from python_fixer.common import resource_path, strict_loads, digest, canonical_bytes
from .plan import validate_plan, expected_keys


def jsonl_records(path: Path, *, hasher=None):
    with path.open("rb") as stream:
        while True:
            line=stream.readline(65537)
            if not line:
                return
            if len(line)>65536 or not line.strip():
                raise ValueError("empty/oversized result row")
            if hasher is not None:
                hasher.update(line)
            yield strict_loads(line.decode("utf-8"))


def _artifact(root: Path, category: str, sha256: str, suffix: str) -> bytes:
    path=root/category/(sha256+suffix)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes evidence root")
    with path.open("rb") as stream:
        data=stream.read(65537)
    if len(data)>65536 or hashlib.sha256(data).hexdigest()!=sha256:
        raise ValueError("missing, oversized or hash-mismatched artifact")
    return data


def validate_records(plan: dict, records, *, evidence_root: Path | None = None) -> list[dict]:
    validate_plan(plan)
    observed=plan["evidence_kind"]=="observed-model"
    if observed and evidence_root is None:
        raise ValueError("observed-model analysis requires candidate and sandbox evidence")
    validator=Draft202012Validator(strict_loads(resource_path("schemas/evaluation-result.schema.json").read_text()))
    sandbox_validator=Draft202012Validator(strict_loads(resource_path("schemas/sandbox-outcome.schema.json").read_text()))
    keys=expected_keys(plan)
    seen=set()
    checkpoints={(a["arm"],c["train_seed"]):c["checkpoint_sha256"] for a in plan["arms"] for c in a["checkpoints"]}
    problems={p["problem_id"]:p for p in plan["problems"]}
    generation_hash=digest(plan["generation"])
    accepted=[]
    for record in records:
        if len(accepted)>=len(keys):
            raise ValueError("extra result beyond complete grid")
        canonical_bytes(record)
        validator.validate(record)
        ints=[record["schema_version"],record["sample_index"],record["generated_tokens"]]
        ints += [record[k] for k in ("train_seed","sampling_seed") if record[k] is not None]
        if any(type(value) is not int for value in ints):
            raise ValueError("integer result fields cannot use floating-point representations")
        key=tuple(record[k] for k in ("arm","train_seed","problem_id","sampling_seed","sample_index"))
        if key not in keys or key in seen:
            raise ValueError("unexpected or duplicate evaluation attempt")
        seen.add(key)
        problem=problems[record["problem_id"]]
        expected={"study_id":plan["study_id"],"model_revision":plan["model_revision"],
            "tokenizer_sha256":plan["tokenizer_sha256"],"dataset_manifest_sha256":plan["dataset_manifest_sha256"],
            "generation_config_sha256":generation_hash,"source_family_id":problem["source_family_id"],
            "inference_profile_sha256":digest(plan["inference_profile"]),
            "checkpoint_sha256":checkpoints[(record["arm"],record["train_seed"])],
            "prompt_sha256":problem["prompt_sha256_by_arm"][record["arm"]]}
        if any(record[k]!=value for k,value in expected.items()):
            raise ValueError("result does not match frozen input/checkpoint identities")
        if record["generated_tokens"]>plan["generation"]["max_new_tokens"] or record["finish_reason"] not in {"eos","length","invalid"}:
            raise ValueError("invalid generation length/finish reason")
        if record["finish_reason"]=="length" and record["generated_tokens"]!=plan["generation"]["max_new_tokens"]:
            raise ValueError("length-terminated generation must exhaust the frozen completion budget")
        if record["all_tests_pass"] and (record["status"]!="OK" or not record["compiled"] or record["regression"] or record["finish_reason"]!="eos"):
            raise ValueError("contradictory full-pass result")
        if record["status"]=="OK" and not record["compiled"]:
            raise ValueError("complete execution requires compiled code")
        if record["status"]=="COMPILE_ERROR" and record["compiled"]:
            raise ValueError("compile-error result cannot claim compiled code")
        if record["status"]!="OK" and record["regression"]:
            raise ValueError("incomplete execution cannot invent an observed per-test regression; conservative rate handles failures")
        regressed=record["regressed_passed_count"]
        if record["status"]=="OK":
            if type(regressed) is not int or not 0<=regressed<=problem["baseline_passed_count"]:
                raise ValueError("complete execution requires bounded per-test regression counts")
            if record["regression"]!=(regressed>0):
                raise ValueError("candidate regression flag disagrees with per-test count")
        elif regressed is not None:
            raise ValueError("incomplete execution cannot invent per-test counts")
        if observed:
            _artifact(evidence_root,"candidates",record["candidate_sha256"],".py")
            raw=_artifact(evidence_root,"sandbox",record["sandbox_report_sha256"],".json")
            report=strict_loads(raw.decode())
            sandbox_validator.validate(report)
            bindings={"candidate_sha256":record["candidate_sha256"],"suite_sha256":problem["test_suite_sha256"],
                      "image_digest":plan["sandbox_image_digest"],"policy_sha256":plan["sandbox_policy_sha256"]}
            if any(report[k]!=value for k,value in bindings.items()):
                raise ValueError("sandbox evidence binding mismatch")
            if (report["status"]!=record["status"] or report["compiled"]!=record["compiled"] or report["all_passed"]!=record["all_tests_pass"]):
                raise ValueError("result contradicts its sandbox evidence")
            if report["total_failed"]!=problem["baseline_failed_count"] or report["total_passed"]!=problem["baseline_passed_count"]:
                raise ValueError("sandbox report omits or changes frozen baseline test counts")
            if report["truncated_output"] and report["status"]!="RESOURCE_LIMIT":
                raise ValueError("truncated output must be a resource failure, never complete execution")
            if report["status"]=="OK":
                from python_fixer.rewards import behavior_reward
                behavior_reward(report,expected_bindings=bindings)  # Semantic count validation, not reward-based evaluation.
                if record["regression"]!=(report["preserved_passed"]<report["total_passed"]):
                    raise ValueError("regression flag contradicts trusted counts")
                if regressed!=report["total_passed"]-report["preserved_passed"]:
                    raise ValueError("per-test regression count contradicts trusted counts")
        accepted.append(record)
    if seen!=keys:
        raise ValueError(f"incomplete evaluation: expected {len(keys)}, observed {len(seen)}; no metrics produced")
    return accepted

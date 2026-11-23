"""Explicit synthetic metadata/results. Never a model or benchmark fixture."""
from python_fixer.common import digest, text_digest


def synthetic_plan(sampled=False):
    arms=[{"arm":"E0","checkpoints":[{"train_seed":None,"checkpoint_sha256":"0"*64}]},
          {"arm":"E1","checkpoints":[{"train_seed":s,"checkpoint_sha256":text_digest("synthetic-checkpoint-"+str(s)),"parent_checkpoint_sha256":"0"*64} for s in (17,29,43)]}]
    return {"schema_version":1,"status":"FROZEN","study_id":"synthetic-fixture-only",
        "study_scope":"synthetic-test","evidence_kind":"synthetic-test","model_revision":"0"*40,
        "tokenizer_sha256":"1"*64,"dataset_manifest_sha256":"2"*64,"enable_thinking":False,"base_checkpoint_sha256":"0"*64,
        "inference_profile":{"precision":"bf16","batch_size":1,"device_model":"synthetic-not-measured","software_lock_sha256":"8"*64},
        "generation":({"mode":"secondary-sampled","do_sample":True,"max_new_tokens":512,"n":2,"k":[1,2],"sampling_seeds":[101,202],"temperature":.7,"top_p":.8,"top_k":20}
                      if sampled else {"mode":"primary-greedy","do_sample":False,"max_new_tokens":512,"n":1,"k":[1],"sampling_seeds":[None]}),
        "arms":arms,"problems":[{"problem_id":f"p{i}","source_family_id":f"f{i//2}","split":"test",
            "bug_category":"boundary" if i%2 else "comparison","difficulty":"easy",
            "baseline_failed_count":1,"baseline_passed_count":1,
            "test_suite_sha256":text_digest("synthetic-suite-"+str(i)),
            "prompt_sha256_by_arm":{"E0":text_digest("synthetic-prompt-"+str(i)),"E1":text_digest("synthetic-prompt-"+str(i))}} for i in range(4)],
        "comparisons":[{"id":"E1-E0","left":"E1","right":"E0","primary":True}],
        "bootstrap":{"replicates":200,"seed":987},"meaningful_gain_absolute":.05,
        "max_regression_increase_absolute":.02,"holm_family_alpha":.05,
        "sandbox_image_digest":"python@sha256:"+"3"*64,"sandbox_policy_sha256":"4"*64}


def synthetic_records(plan):
    rows=[]
    for arm in plan["arms"]:
        for checkpoint in arm["checkpoints"]:
            for problem in plan["problems"]:
                for sampling_seed in plan["generation"]["sampling_seeds"]:
                    for index in range(plan["generation"]["n"]):
                        success=arm["arm"]=="E1" and index==0
                        rows.append({"schema_version":1,"study_id":plan["study_id"],"arm":arm["arm"],
                            "train_seed":checkpoint["train_seed"],"checkpoint_sha256":checkpoint["checkpoint_sha256"],
                            "problem_id":problem["problem_id"],"source_family_id":problem["source_family_id"],
                            "model_revision":plan["model_revision"],"tokenizer_sha256":plan["tokenizer_sha256"],
                            "dataset_manifest_sha256":plan["dataset_manifest_sha256"],
                            "prompt_sha256":problem["prompt_sha256_by_arm"][arm["arm"]],
                            "generation_config_sha256":digest(plan["generation"]),"inference_profile_sha256":digest(plan["inference_profile"]),
                            "candidate_sha256":text_digest("synthetic-candidate"),"sandbox_report_sha256":"7"*64,
                            "sampling_seed":sampling_seed,"sample_index":index,"finish_reason":"eos",
                            "generated_tokens":20,"generation_ms":10.0,"status":"OK","compiled":True,
                            "all_tests_pass":success,"regression":False,"regressed_passed_count":0})
    return rows

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from python_fixer.common import canonical_bytes, digest, write_json
from python_fixer.evaluation.plan import validate_plan, expected_keys
from python_fixer.evaluation.records import validate_records, jsonl_records
from python_fixer.evaluation.run import main
from m2_fixtures import synthetic_plan, synthetic_records


class PlanAndRecordTests(unittest.TestCase):
    def setUp(self):
        self.plan=synthetic_plan()
        self.rows=synthetic_records(self.plan)

    def test_full_expected_grid(self):
        validate_plan(self.plan)
        self.assertEqual(len(expected_keys(self.plan)),16)
        self.assertEqual(len(validate_records(self.plan,self.rows)),16)

    def test_draft_plan_and_duplicate_arms_rejected(self):
        self.plan["status"]="DRAFT"
        with self.assertRaisesRegex(ValueError,"frozen"):
            validate_plan(self.plan)
        self.plan["status"]="FROZEN"
        self.plan["arms"].append(copy.deepcopy(self.plan["arms"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate arm"):
            validate_plan(self.plan)

    def test_missing_duplicate_and_extra_results_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],[self.rows[0]]+self.rows[:-1]):
            with self.subTest(length=len(rows)),self.assertRaises(ValueError):
                validate_records(self.plan,rows)

    def test_altered_checkpoint_prompt_family_or_generation_rejected(self):
        for key,value in (("checkpoint_sha256","a"*64),("prompt_sha256","a"*64),
                          ("source_family_id","different-family"),("generation_config_sha256","a"*64),
                          ("inference_profile_sha256","a"*64)):
            rows=copy.deepcopy(self.rows)
            rows[0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"frozen"):
                validate_records(self.plan,rows)

    def test_failed_or_truncated_output_cannot_claim_pass(self):
        for overrides in ({"status":"TIMEOUT","all_tests_pass":True},
                          {"finish_reason":"length","generated_tokens":512,"all_tests_pass":True},
                          {"compiled":False,"all_tests_pass":True},
                          {"regression":True,"all_tests_pass":True}):
            rows=copy.deepcopy(self.rows)
            rows[0].update(overrides)
            with self.assertRaises(ValueError):
                validate_records(self.plan,rows)

    def test_stronger_prompt_baseline_cannot_use_different_base(self):
        self.plan["arms"].append({"arm":"E0-P","checkpoints":[{"train_seed":None,"checkpoint_sha256":"9"*64}]})
        for problem in self.plan["problems"]:
            problem["prompt_sha256_by_arm"]["E0-P"]="5"*64
        with self.assertRaisesRegex(ValueError,"exact base"):
            validate_plan(self.plan)

    def test_synthetic_data_cannot_be_confirmatory(self):
        self.plan["study_scope"]="confirmatory"
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def test_observed_model_requires_artifacts(self):
        self.plan.update(evidence_kind="observed-model",study_scope="exploratory")
        with self.assertRaisesRegex(ValueError,"requires candidate"):
            validate_records(self.plan,self.rows)

    def test_sft_parent_relation_is_required_for_e2(self):
        self.plan["arms"].append({"arm":"E2","checkpoints":[{"train_seed":s,"checkpoint_sha256":"9"*64,
             "parent_checkpoint_sha256":"8"*64} for s in (17,29,43)]})
        for problem in self.plan["problems"]:
            problem["prompt_sha256_by_arm"]["E2"]=problem["prompt_sha256_by_arm"]["E1"]
        with self.assertRaisesRegex(ValueError,"parent"):
            validate_plan(self.plan)

    def test_direct_grpo_and_sft_must_start_from_base(self):
        self.plan["arms"].append({"arm":"E3","checkpoints":[{"train_seed":s,"checkpoint_sha256":"9"*64,
            "parent_checkpoint_sha256":self.plan["arms"][1]["checkpoints"][0]["checkpoint_sha256"]} for s in (17,29,43)]})
        for problem in self.plan["problems"]:
            problem["prompt_sha256_by_arm"]["E3"]=problem["prompt_sha256_by_arm"]["E1"]
        with self.assertRaisesRegex(ValueError,"base initializer"):
            validate_plan(self.plan)

    def test_float_representation_of_integer_protocol_fields_rejected(self):
        self.plan["generation"]["n"]=1.0
        with self.assertRaisesRegex(ValueError,"integer protocol"):
            validate_plan(self.plan)
        self.plan["generation"]["n"]=1
        self.rows[0]["sample_index"]=0.0
        with self.assertRaisesRegex(ValueError,"integer result"):
            validate_records(self.plan,self.rows)
        for checkpoint in self.plan["arms"][-1]["checkpoints"]:
            checkpoint["parent_checkpoint_sha256"]="0"*64
        validate_plan(self.plan)
        self.plan["arms"][1]["checkpoints"][0]["parent_checkpoint_sha256"]="9"*64
        with self.assertRaisesRegex(ValueError,"base initializer"):
            validate_plan(self.plan)


class ObservedEvidenceIntegrityTests(unittest.TestCase):
    """Synthetic files emulate the recorded-data interface; no model is run."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=synthetic_plan()
        self.plan.update(evidence_kind="observed-model",study_scope="exploratory")
        self.rows=synthetic_records(self.plan)
        problems={p["problem_id"]:p for p in self.plan["problems"]}
        source=b"# synthetic integrity fixture; not a model output\n"
        candidate_sha=hashlib.sha256(source).hexdigest()
        (self.root/"candidates").mkdir()
        (self.root/"candidates"/(candidate_sha+".py")).write_bytes(source)
        (self.root/"sandbox").mkdir()
        for index,row in enumerate(self.rows):
            row["candidate_sha256"]=candidate_sha
            report={"schema_version":1,"request_id":"synthetic-request-"+str(index),
                "candidate_sha256":candidate_sha,"suite_sha256":problems[row["problem_id"]]["test_suite_sha256"],
                "image_digest":self.plan["sandbox_image_digest"],"policy_sha256":self.plan["sandbox_policy_sha256"],
                "status":"OK","compiled":True,"fixed_failed":int(row["all_tests_pass"]),"total_failed":1,
                "preserved_passed":1,"total_passed":1,"all_passed":row["all_tests_pass"],"wall_ms":1,
                "cpu_ms":1,"peak_memory_bytes":1,"truncated_output":False,"runner_version":"synthetic-double","diagnostics":""}
            raw=canonical_bytes(report)+b"\n"
            row["sandbox_report_sha256"]=hashlib.sha256(raw).hexdigest()
            (self.root/"sandbox"/(row["sandbox_report_sha256"]+".json")).write_bytes(raw)

    def test_bound_recorded_files_validate_integrity_only(self):
        self.assertEqual(len(validate_records(self.plan,self.rows,evidence_root=self.root)),16)

    def test_changed_artifact_bytes_rejected(self):
        path=self.root/"sandbox"/(self.rows[0]["sandbox_report_sha256"]+".json")
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError,"hash-mismatched"):
            validate_records(self.plan,self.rows,evidence_root=self.root)

    def test_valid_hash_wrong_suite_rejected(self):
        row=self.rows[0]
        old=self.root/"sandbox"/(row["sandbox_report_sha256"]+".json")
        report=json.loads(old.read_text())
        report["suite_sha256"]="9"*64
        raw=canonical_bytes(report)+b"\n"
        row["sandbox_report_sha256"]=hashlib.sha256(raw).hexdigest()
        (self.root/"sandbox"/(row["sandbox_report_sha256"]+".json")).write_bytes(raw)
        with self.assertRaisesRegex(ValueError,"binding mismatch"):
            validate_records(self.plan,self.rows,evidence_root=self.root)

    def rewrite_report(self,row,changes):
        old=self.root/"sandbox"/(row["sandbox_report_sha256"]+".json")
        report=json.loads(old.read_text())
        report.update(changes)
        raw=canonical_bytes(report)+b"\n"
        row["sandbox_report_sha256"]=hashlib.sha256(raw).hexdigest()
        (self.root/"sandbox"/(row["sandbox_report_sha256"]+".json")).write_bytes(raw)

    def test_truncated_full_pass_report_rejected_even_with_valid_file_hash(self):
        row=next(r for r in self.rows if r["all_tests_pass"])
        self.rewrite_report(row,{"truncated_output":True})
        with self.assertRaisesRegex(ValueError,"truncated"):
            validate_records(self.plan,self.rows,evidence_root=self.root)

    def test_omitted_test_denominators_rejected(self):
        row=self.rows[0]
        self.rewrite_report(row,{"total_failed":2})
        with self.assertRaisesRegex(ValueError,"baseline test counts"):
            validate_records(self.plan,self.rows,evidence_root=self.root)


class AnalysisCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=synthetic_plan()
        self.rows=synthetic_records(self.plan)
        write_json(self.root/"plan.json",self.plan)
        (self.root/"results.jsonl").write_bytes(b"".join(canonical_bytes(r)+b"\n" for r in self.rows))

    def args(self,action="analyze"):
        return [action,"--plan",str(self.root/"plan.json"),"--plan-sha256",digest(self.plan),
                "--results",str(self.root/"results.jsonl"),"--output",str(self.root/"analysis.json")]

    def test_complete_cli_and_exact_file_hash(self):
        self.assertEqual(main(self.args()),0)
        report=json.loads((self.root/"analysis.json").read_text())
        self.assertEqual(report["results_file_sha256"],hashlib.sha256((self.root/"results.jsonl").read_bytes()).hexdigest())
        self.assertEqual(report["evidence_kind"],"synthetic-test")
        self.assertEqual(report["candidate_count"],16)

    def test_validation_only_and_refuse_overwrite(self):
        self.assertEqual(main(self.args("validate")),0)
        original=(self.root/"analysis.json").read_bytes()
        self.assertEqual(main(self.args()),3)
        self.assertEqual((self.root/"analysis.json").read_bytes(),original)

    def test_bad_plan_pin_and_incomplete_results_do_not_publish(self):
        args=self.args()
        args[args.index("--plan-sha256")+1]="0"*64
        self.assertEqual(main(args),3)
        self.assertFalse((self.root/"analysis.json").exists())
        (self.root/"results.jsonl").write_text("")
        self.assertEqual(main(self.args()),3)
        self.assertFalse((self.root/"analysis.json").exists())

    def test_jsonl_duplicate_keys_and_nonfinite_numbers_rejected(self):
        for raw in ('{"a":1,"a":2}\n','{"a":NaN}\n'):
            (self.root/"bad.jsonl").write_text(raw)
            with self.assertRaises(ValueError):
                list(jsonl_records(self.root/"bad.jsonl"))


if __name__=="__main__":
    unittest.main()

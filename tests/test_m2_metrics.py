import unittest
import copy
from python_fixer.common import text_digest
from python_fixer.evaluation.plan import validate_plan
from python_fixer.evaluation.metrics import pass_at_k, quantile, summarize
from python_fixer.evaluation.statistics import holm_adjust, paired_cluster_analysis
from python_fixer.evaluation.records import validate_records
from python_fixer.evaluation.analysis import analyze_validated
from m2_fixtures import synthetic_plan, synthetic_records


class MetricTests(unittest.TestCase):
    def test_unbiased_pass_at_k_known_vectors(self):
        self.assertEqual(pass_at_k(20,0,10),0)
        self.assertEqual(pass_at_k(20,20,1),1)
        self.assertAlmostEqual(pass_at_k(10,1,1),.1)
        self.assertAlmostEqual(pass_at_k(10,1,5),.5)
        self.assertEqual(pass_at_k(10,6,5),1)
        self.assertAlmostEqual(pass_at_k(4,2,2),5/6)

    def test_invalid_counts_and_boolean_rejected(self):
        for vector in ((0,0,1),(3,4,1),(3,1,4),(3,1,0),(True,0,1)):
            with self.subTest(vector=vector),self.assertRaises(ValueError):
                pass_at_k(*vector)

    def test_quantile_linear_interpolation(self):
        self.assertEqual(quantile([1,4,2,3],.5),2.5)
        self.assertEqual(quantile([1],.95),1)

    def test_sample_repeats_and_checkpoint_seeds_not_pooled(self):
        plan=synthetic_plan(sampled=True)
        report=analyze_validated(plan,validate_records(plan,synthetic_records(plan)))
        # Each policy generates one success out of n=2, hence its pass@2=1.
        # Pooling different policy checkpoints would incorrectly yield <1.
        self.assertEqual(report["arms"]["E1"]["metrics"]["pass@2"],1)
        self.assertEqual(report["arms"]["E1"]["metrics"]["pass@1"],.5)
        self.assertEqual(report["arms"]["E1"]["problem_count"],4)
        self.assertEqual(report["arms"]["E1"]["seed_count"],3)

    def test_synthetic_success_never_claims_confirmatory_improvement(self):
        plan=synthetic_plan()
        report=analyze_validated(plan,validate_records(plan,synthetic_records(plan)))
        result=report["comparisons"]["E1-E0"]
        self.assertEqual(result["difference"],1)
        self.assertFalse(result["preregistered_numeric_criteria_met"])
        self.assertFalse(result["confirmatory_criteria_applicable"])
        self.assertEqual(report["independent_run_authentication"],"NOT_VERIFIED_BY_ANALYSIS")

    def test_incomplete_execution_has_conservative_regression_bound(self):
        plan=synthetic_plan()
        rows=synthetic_records(plan)
        bad=next(r for r in rows if r["arm"]=="E0")
        bad.update(status="TIMEOUT",all_tests_pass=False,regression=False,regressed_passed_count=None)
        valid=validate_records(plan,rows)
        report=analyze_validated(plan,valid)["arms"]["E0"]
        self.assertEqual(report["observed_regression_rate"],0)
        self.assertEqual(report["conservative_regression_rate"],.25)

    def test_confirmatory_numeric_decision_branch_with_explicit_test_double(self):
        # Exercise pure decision logic on synthetic in-memory values. This deliberately
        # does not call the observed-record loader and cannot publish a genuine report.
        plan=synthetic_plan()
        plan.update(study_scope="confirmatory",evidence_kind="observed-model",bootstrap={"replicates":10000,"seed":987})
        e1={c["train_seed"]:c["checkpoint_sha256"] for c in plan["arms"][1]["checkpoints"]}
        plan["arms"] += [
            {"arm":"E0-P","checkpoints":[{"train_seed":None,"checkpoint_sha256":"0"*64}]},
            {"arm":"E2","checkpoints":[{"train_seed":s,"checkpoint_sha256":text_digest("test-e2-"+str(s)),"parent_checkpoint_sha256":e1[s]} for s in (17,29,43)]},
            {"arm":"E3","checkpoints":[{"train_seed":s,"checkpoint_sha256":text_digest("test-e3-"+str(s)),"parent_checkpoint_sha256":"0"*64} for s in (17,29,43)]}]
        original=plan["problems"][0]
        plan["problems"]=[]
        for i in range(12):
            problem=copy.deepcopy(original)
            problem.update(problem_id=f"unit-p{i}",source_family_id=f"unit-f{i}")
            prompt=text_digest("unit-original-"+str(i))
            problem["prompt_sha256_by_arm"]={a["arm"]:(text_digest("unit-stronger-"+str(i)) if a["arm"]=="E0-P" else prompt) for a in plan["arms"]}
            plan["problems"].append(problem)
        plan["comparisons"].append({"id":"E2-E1","left":"E2","right":"E1","primary":True})
        validate_plan(plan)
        rows=synthetic_records(plan)
        for row in rows:
            row["all_tests_pass"]=row["arm"] in {"E1","E2","E3"}
        report=analyze_validated(plan,rows)
        self.assertTrue(report["comparisons"]["E1-E0"]["preregistered_numeric_criteria_met"])
        self.assertFalse(report["comparisons"]["E2-E1"]["preregistered_numeric_criteria_met"])
        self.assertEqual(report["independent_run_authentication"],"NOT_VERIFIED_BY_ANALYSIS")


class StatisticalTests(unittest.TestCase):
    def test_holm_monotonic_adjustment(self):
        self.assertEqual(holm_adjust({"a":.01,"b":.04,"c":.03}),{"a":.03,"c":.06,"b":.06})

    def test_holm_invalid_values(self):
        for values in ({},{"a":float("nan")},{"a":-1},{"a":True}):
            with self.assertRaises(ValueError):
                holm_adjust(values)

    def test_all_zero_difference_has_p_one(self):
        result=paired_cluster_analysis({"f1":{"17":[0,0]},"f2":{"17":[0]}},replicates=100)
        self.assertEqual(result["difference"],0)
        self.assertEqual(result["permutation_p"],1)
        self.assertEqual(result["ci95"],[0,0])

    def test_exact_paired_sign_flip_known_probability(self):
        result=paired_cluster_analysis({f"f{i}":{"17":[1]} for i in range(8)},replicates=100)
        self.assertTrue(result["permutation_exact"])
        self.assertEqual(result["permutation_count"],256)
        self.assertEqual(result["permutation_p"],2/256)

    def test_weighted_problem_mean_not_unweighted_family_mean(self):
        result=paired_cluster_analysis({"large":{"17":[1,1,1]},"small":{"17":[-1]}},replicates=100)
        self.assertEqual(result["difference"],.5)

    def test_reordering_input_does_not_change_seeded_result(self):
        a={"f1":{"29":[1,0],"17":[0,0]},"f2":{"17":[1],"29":[0]}}
        b=dict(reversed(list(a.items())))
        self.assertEqual(paired_cluster_analysis(a,replicates=100),paired_cluster_analysis(b,replicates=100))

    def test_invalid_seed_pairing_and_lengths_rejected(self):
        for bad in ({"f1":{"17":[0]},"f2":{"29":[0]}},{"f1":{"17":[0],"29":[0,1]}}):
            with self.assertRaises(ValueError):
                paired_cluster_analysis(bad,replicates=100)

    def test_monte_carlo_path_is_seeded_and_never_reports_zero_p(self):
        differences={f"f{i}":{"17":[1]} for i in range(17)}
        first=paired_cluster_analysis(differences,replicates=100)
        second=paired_cluster_analysis(differences,replicates=100)
        self.assertEqual(first,second)
        self.assertFalse(first["permutation_exact"])
        self.assertGreater(first["permutation_p"],0)

    def test_seed_resampling_retains_training_variation(self):
        result=paired_cluster_analysis({"f1":{"17":[0],"29":[1]},"f2":{"17":[0],"29":[1]}},replicates=100)
        self.assertEqual(result["difference"],.5)
        self.assertEqual(result["ci95"],[0,1])


if __name__=="__main__":
    unittest.main()

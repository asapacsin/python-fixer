# 02 — Dataset construction and leakage control

## Sources and licensing

Primary: authored pure functions with property-based and example tests, then
deterministic AST mutations. Preserve authorship/license and mutation seed.
QuixBugs is a small external diagnostic (40 algorithm problems, with Java/Python
variants); audit per-file license and dependencies at the selected commit. Its
public popularity makes contamination likely and it cannot supply the entire
heldout. BugsInPy is a repository/environment benchmark, not drop-in pure-function
JSONL: optional later external-validity track only after reproducing original
environments and auditing upstream project licenses. Do not silently turn complex
bugs into artificial function tasks and report the original benchmark name.

Additional permissively licensed function corpora require source URL, exact commit,
path, original author/license and redistribution decision. Unknown licenses are
quarantined. Acquisition manifests contain HTTP/git source, retrieval date,
license file hash, content SHA-256 and transformation version. A repository-level
license alone does not establish every upstream code file's license. Generated
problems require generator provenance and review; do not count generated tests
as independent semantic authority without checking their intended behavior.

## Build order

Acquire -> license gate -> normalize -> group related functions -> split families
-> generate mutations -> sandbox-verify -> deduplicate -> freeze manifests.
Use tokenizer/AST normalization that preserves semantic literals and identifiers
for the canonical function hash. Group near-duplicate AST/token clones,
renamings, common ancestor commits and known translations into `source_family_id`.
All revisions and mutations of a family stay in one split. If a cross-split clone
is found later, quarantine the entire family and regenerate/version the split;
never move only an offending test row into training.

Use deterministic group hash assignment with a recorded seed and initial
80/10/10 train/validation/test ratios. Adjust family allocation before sealing
to reach evaluation target counts and bug/difficulty coverage, documenting the
selection algorithm. No performance-based inclusion or exclusion. Report both
problem count and independent family count. Use one held-out mutation per function
initially; additional variants remain clustered for analysis.

## Row contract v1

The machine-readable shape is [problem.schema.json](../schemas/problem.schema.json);
[problem.jsonl](../examples/problem.jsonl) is an authored illustrative row, not
a sandbox-verified dataset. Each private canonical row includes:

| Field | Meaning |
|---|---|
| `problem_id`, `source_function_id`, `source_family_id` | Stable unique IDs and grouping |
| `split`, `schema_version` | Frozen split and schema |
| `signature`, `behavior`, `buggy_code`, `reference_code` | Task and reference source |
| `bug_category`, `difficulty` | Controlled labels, assigned without model scores |
| `provenance` | source URI, revision, path, license, content hash, mutation metadata |
| `test_suite_id`, `test_suite_sha256`, `tests` | Immutable sandbox test bundle |
| `verification` | `NOT_RUN` or verified counts, image/policy hashes and evidence |

Tests are callable invocations with JSON arguments, keyword arguments and expected
JSON values, using a declared comparator (exact initially). M1 adds trusted
runner-owned property checks and edge cases, serialized in a separate versioned
test format. No test code from model outputs. This initial function subset does
not support arbitrary classes, NaN equality or custom objects; extend the schema
and comparator through a reviewed version change, not implicit coercion.

## Verification gates

Run references in the approved sandbox: every test must pass; buggy source must
compile and fail >=1 semantic assertion, with >=1 previously passing regression
test. Require at least four substantive tests covering boundaries, ordinary
inputs and relevant invariants. Repeat reference/buggy runs three times; quarantine
flaky or resource-dependent cases. A syntax-error mutation is not useful repair
evidence for the primary track. Discard equivalent/no-op mutations, output-only
or test-mutating changes and cases with conflicting behavioral descriptions.
Record which baseline tests fail/pass and test count before policy generation.
Analyze mutation survival, attrition by category and manual review of at least
50 randomly sampled accepted cases before scaling. Difficulty labels use fixed
human rules (one local edit / interacting branches / algorithmic change), with
review notes; never derive labels from held-out model performance.

## Views and isolation

Private canonical rows never enter the model trainer wholesale. SFT view: prompt
(behavior, signature, buggy source) + corrected source; no tests, IDs/provenance
strings leaking answer location, verification data or held-out references.
RL view: prompt + train problem ID; only the reward service can map it to train
tests/reference. Evaluation view for generation: prompt + opaque problem ID only.
Sealed evaluator alone has test references. Validation reward tests are not RL
rewards. Public demonstration JSONL is separately authored and never a heldout.

Do not initially include visible tests in prompts; this keeps reward/evaluation
tests private. A visible-test experiment requires separate disjoint public and
private tests and is an explicit later ablation. Random splits after mutation,
substring dedup alone and mixing QuixBugs into train/test are prohibited.

## Manifest and storage

Manifest records schema/version, all source revisions, builder commit, grouping
algorithm/seed, group membership hashes, row/test bundle hashes, counts by split,
category/difficulty, exclusion reasons and license decisions. Sort rows by stable
ID, UTF-8 LF JSONL; hash exact bytes. Store corpus and test bundles externally.
Freeze a manifest before E0; revisions create a new study namespace. Schema
validation, duplicate IDs, split collision, clone-family tests and a golden
deterministic rebuild are required M1 tests.

After pinning the tokenizer in M2, apply the same token-eligibility rule to all
arms/splits before the final study manifest: templated prompt <=1536 tokens and
reference completion including EOS <=512 tokens. Record excluded families and
counts. M1 may produce a preliminary CPU-verified manifest; M2's token-filtered
manifest is the frozen study input used by every arm. Do not silently truncate
SFT targets or select eligibility from model success rates. `source_sha256`
hashes normalized pre-mutation reference function bytes; retain the upstream
blob hash in acquisition metadata separately.

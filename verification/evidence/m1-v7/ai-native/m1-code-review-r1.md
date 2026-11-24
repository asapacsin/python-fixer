<!-- independent agent reviewer (fresh Opus 5.5 agent, high effort, read-only; user-designated; not human). Commits 4a4a001 c1e41da 156bded -->
# Independent read-only review: 4a4a001, c1e41da, 156bded (base 9356e6b)

## Verdict: REWORK

This covers the three commits and the evidence files listed below. I did not judge M1 as a whole. The roadmap and docs say M1 is not complete, and they are right: the manual 50-case review and the independent review are still open.

- **4a4a001 (extraction):** PASS, with minor findings. It is authorized by `verification/contracts/m2-cpu.json`.
- **c1e41da (sandbox):** PASS on the security question. The change is justified and still bounded, and it does not weaken the candidate limit. Its doc and roadmap claims need rework (findings 3 and 7).
- **156bded (data):** REWORK. The audit's leakage checks can never fail, so they prove nothing. Provenance falls back silently to the wrong value. Split membership shifts between dataset versions without being documented.

I modified no files and changed no git state; the working tree was clean. Both gates passed when I ran them (this was before your note that Docker is down; none of these unit tests need Docker):
- `unittest discover -s tests`: 97 tests, OK.
- `validate_repository.py`: PASS.

## What the evidence checks confirmed

- **First run (`m1-evidence-20261123.json`):** its `working_diff_sha256` equals `git diff 4a4a001 c1e41da -- configs src tests`. So the 22/22 preflight ran on the code that was later committed as c1e41da.
- **Policy hash:** `Policy.load(configs/sandbox.json).sha256` at HEAD gives `7740fea4…48b`, which matches the preflight evidence.
- **v2 run (`m1-evidence-v2-20261123.json`):** its `working_diff_sha256` equals `git diff c1e41da 156bded --diff-filter=M`, and all five untracked-file hashes match the committed blobs. So the v2 build is bound to the 156bded content.
- **v1 plan:** the current v1 plan reproduces `content_plan_sha256` of `artifacts/data/m1-approved-v1`, so the claim that v1 is unchanged holds.
- **v2 dataset:** the manifest digest is `230a246f…` and matches the pin. All 12 rejections are mutants; none carries the `authored-bug-v1` id suffix. The counts 74/12, train/validation/test 60/10/4 and 48 families match the docs.

## Findings, by severity

**1. Medium-High: the audit's leakage checks can never fail.** (`src/python_fixer/data/audit.py:54-71`)
- **Defect:** `rendered = str(view)` produces a Python repr, which turns each newline into a literal `\n`. So `row["reference_code"] not in rendered` is always true for any multi-line reference. The `digest(t)` / `str(t)` test-record checks are equally ineffective.
- **Failure scenario:** I patched `model_view` to return a prompt that embeds the full `reference_code` plus `json.dumps(row["tests"])`. All four `evaluation_view_*` checks still returned True. The "mechanical 50/50 PASS" in `docs/m1-data-audit.md` therefore includes three checks that cannot fail. Production views are protected by the existing `test_projection_drops_private_fields`, so this is misleading evidence, not a live leak.
- **Fix:**
  - Check the raw strings: walk the prompt messages and look for the reference, and for each test's expected value and canonical JSON.
  - Add negative tests that inject a leak and assert the check fails.
  - Re-run the audit and correct the 50/50 claim.

**2. Medium: provenance and source hash fall back silently to v1.** (`src/python_fixer/data/build.py:36-50, 81, 99`)
- **Defect:** `plan_rows(..., source="authored-fixtures-v1")` takes its `seeds` separately from `source`. Paths are looked up with `paths.get(function, "…/fixtures.json")`.
- **Failure scenario:** `plan_rows(fixture_seeds("authored-fixtures-v2"))` without `source=` labels `fixture-gcd-v2` as revision `fixtures-v1`, path `fixtures.json`, with no error. I reproduced this. `write_dataset` likewise hashes `fixture_seeds(config["source"])` rather than the seeds actually planned, so the manifest's `source_sha256` can describe different input.
- **Fix:**
  - Raise when a seed id is missing from `seed_paths(source)`.
  - Or derive the seeds from `source` inside `plan_rows`.
  - Record the digest of the seeds that were actually used.

**3. Medium: the security suite is marked done without full real-runtime coverage.** (`docs/implementation-roadmap.md:36`)
- **Defect:** "[x] Pass the full security suite on an enforcing runsc worker (22/22)" overstates what ran. The 22 real-runsc checks in `sandbox/preflight.py` do not cover several rows of the spec 08 attack table:
  - DNS (only a TCP connection to 169.254.169.254 is probed);
  - path traversal, symlinks and oversized nested JSON;
  - worker crash, stale cache and missing tests;
  - reading expected answers or mutating the assertion library inside the runtime;
  - surviving sleeping descendants (the fork probe kills its own children).
- **Why it matters:** the M1 contract's acceptance item is "All security attack cases in spec08". Some of these rows are covered by unit tests using doubles, and the repo's own docs say those doubles "do not certify containment".
- **Fix:** map each spec 08 row to a real-runsc check or a documented gap, and uncheck or qualify the roadmap item.

**4. Medium: the evidence behind the pids-limit change was not kept.** (spec 08 lines 28-32; `docs/m1-implementation.md:33-36`)
- **Defect:** the rationale says `--pids-limit=32` killed the gVisor Sentry (the runsc kernel process) on a 4-child fork storm, and that 64 was insufficient. Neither failing run is in `.ai-native/`; the only other preflight file there is a BLOCKED "no runsc" report. AGENTS rule 9 says to keep negative results, and the contract requires a recorded rationale for spec changes.
- **Failure scenario:** a reviewer cannot confirm that 128 is needed, or how much headroom it has. On another runsc version, platform or kernel, a fork storm could exceed 128 host pids and produce INTERNAL_ERROR again. INTERNAL_ERROR means "no reward, retry", so a hostile candidate could trigger that path deliberately. The preflight `process-cap` check would catch such a regression, which is the right mitigation.
- **Fix:**
  - Retain the failing reports.
  - Have the preflight record the number of children reached and the cgroup's peak pids/events count.
  - Add a thread-storm probe (`_thread`/`threading`, bypassing the import filter like the other probes). Thread accounting under gVisor RLIMIT_NPROC is untested.

**5. Medium-Low: splits shift between dataset versions, undocumented.** (data build, families)
- **Defect:** adding v2 seeds changed family membership. `fixture-count_equal-v1` and `fixture-count_positive-v1` were train in v1 and are validation in v2, because `count_in_range-v2` joined their family. Spec 02 treats a revision as a new study namespace, so this is not a spec violation.
- **Failure scenario:** anything trained or selected on v1 train and then validated on v2 is contaminated. The docs never mention the shift.
- **Fix:** record the split diff between versions in the manifest or docs, and add a guard that rejects mixing namespaces.

**6. Low-Medium: attrition counts use two category systems.** (`src/python_fixer/data/audit.py:78-84`)
- **Defect:** accepted mutants are now categorized by the construct they edit (`build.py:71`), but rejected mutants are counted under the seed's authored category.
- **Failure scenario:** "attrition by category" (spec 02) and mutation survival per category cannot be compared. The code's `note` admits this rather than fixing it.
- **Fix:** store `bug_category` and `mutation_operator` in the manifest's `rejected` entries and count by those.

**7. Low: overstated or stale docs.**
- `docs/m1-implementation.md:69,72` cites `.ai-native/m1-sandbox-observed-22.json` and `m1-current-blocker.json`; neither exists. "Current preflight returns exit3 BLOCKED" is also stale.
- `docs/m1-implementation.md:10` still says "16 … fixture seeds".
- `docs/m2-cpu-analysis.md:4` still says "M1 sandbox security remains blocked".
- `docs/m1-data-audit.md:18` calls a same-author rebuild "independent". No rebuild artifact is retained, so the determinism claim is unverified.
- The "Author findings (fixed before freezing)" section shows that tests and bugs were edited after reading sampled cases and sandbox outcomes. The final 50-case sample is therefore not an untouched estimate. The doc should state this, and the pre-fix build or worksheet should be kept as a negative result.

**8. Low: the policy allows more than the spec says.** (`src/python_fixer/sandbox/policy.py:32`)
- **Defect:** `self.pids < self.runtime_pids <= 256` accepts values from 33 to 256, but spec 08 states 128.
- **Failure scenario:** a config can raise the limit to 256, and validation still passes with no spec change.
- **Fix:** pin it to 128, or write the allowed range into spec 08.
- **Related:** `preflight.py` is not part of the hash input for `Policy.sha256`, so the probes can change while the policy hash stays the same. This predates these commits.

**9. Low: mutator robustness.** (`src/python_fixer/data/mutations.py:21-23, 67`)
- **Defect:** `str.splitlines` also splits on `\f`, `\v`, `\x1c`–`\x1e`, `\x85`, `\u2028` and `\u2029`, but AST line numbers only count `\n`, `\r\n` and `\r`.
- **Failure scenario:** a seed containing one of those characters gets misaligned offsets. The mutant is then silently dropped, or `decode` / `ast.parse` raises an uncaught exception during the build. Risk is low because the seeds are trusted. The `ast.dump` equality guard is otherwise strong.
- **Fix:** compute line offsets with universal-newline semantics.

**10. Low: extraction details.** (`src/python_fixer/evaluation/extraction.py:16, 40, 50`)
- The ban on "```" and "~~~" applies anywhere in the text, so a valid module with these inside a string or comment is marked INVALID.
- Prose is not detected. It shows up later as a worker COMPILE_ERROR, which affects how compile-failure metrics should be read.
- Other Qwen control tokens such as `<tool_call>` and `<|fim_*|>` are not rejected.
- The docs say "never raise", but the function raises `TypeError` for input that is not str or bytes.
- The module is not wired into anything yet, which is consistent with the roadmap's M2 item staying unchecked.

## Judgment on c1e41da's security change

It is justified and bounded:
- The 32-process cap on the candidate stays in place as RLIMIT_NPROC inside gVisor.
- The real `process-cap` probe passed, which requires fork to fail before 32 children.
- The host pids bound only applies to each sandbox, is lifted to 128, and is included in the policy hash.
- The unit test rejects 32/32, 32/257 and 33/128.

The weak points are findings 4 and 8: no retained evidence or headroom measurement, no thread probe, and a loose upper bound in the code.

## Not verified

- I re-ran no sandbox or preflight, per your instructions and because Docker is down. Every runtime claim rests on the `.ai-native` JSON, which carries no timestamp or HEAD in the preflight file itself.
- The "64 insufficient" observation and the pids-limit=32 failure.
- Rebuild determinism of v2, since there is no second artifact.
- Whether the fixture behavior texts are good and their tests are adequate. I only did a spot-read of the bug diffs; for example, the `is_palindrome` bug `s[::1]` is labelled "operator".
- How gVisor RLIMIT_NPROC counts threads.
- Hosted CI, the dedicated worker and the mTLS deployment.
- The manual 50-case review, which is still pending.
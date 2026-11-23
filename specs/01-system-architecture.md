# 01 — System architecture and interfaces

Decision: a modular Python batch pipeline on one GPU, with a separate disposable
CPU execution worker and a separately permissioned held-out evaluator. Use TRL
GRPOTrainer, not a custom policy-gradient implementation. See
[ADR-001](../docs/adr/001-training-and-execution.md) for options and reversal triggers.

## Ownership and data flow

1. Data builder acquires licensed inputs, normalizes source families, splits
   families, mutates only training families and requests sandbox verification.
   It writes immutable manifests and training/validation/public metadata views.
2. SFT trainer reads only train pairs and validation; stores adapters, complete
   resume checkpoints and logs under a persistent run root.
3. GRPO trainer owns generation, token log probabilities and optimizer state.
   A reward adapter sends source text and train problem IDs to the CPU worker.
   It receives trusted outcomes and computes scalar rewards. Candidate text
   never becomes host code, a shell command or a path.
4. Execution gateway owns immutable test lookup, candidate parsing, policy,
   resource limits, retries and outcome signatures/hashes. Train credentials
   cannot select validation/test problem IDs. Every candidate gets a new sandbox.
5. Independent evaluator reads frozen policy artifacts and sealed test storage.
   It generates candidates without test feedback and records every attempt.
   Training workers cannot access evaluator tests, references or credentials.
6. Artifact manager writes atomic local checkpoints, hashes them and synchronizes
   immutable run objects to persistent storage. The report consumes manifests,
   never a trainer's informal success string.

No workflow framework or database is needed initially. JSONL, content-addressed
files and atomic JSON manifests are sufficient for a single node and four CPU
workers. File ownership and process roles do not replace OS and storage permissions.

## Planned package boundaries (M1 onward)

Use `src/python_fixer/{data,sandbox,rewards,evaluation,training}` and matching
tests as needed. Do not populate empty packages now. CLI entrypoints are planned:

| Milestone | Planned command | Responsibility |
|---|---|---|
| M1 | `python -m python_fixer.data.build --config ...` | Deterministic build + manifest |
| M1 | `python -m python_fixer.sandbox.preflight --config ...` | Isolation and attack checks |
| M2 | `python -m python_fixer.evaluation.run --config ...` | Frozen candidate generation / scoring |
| M3 | `python -m python_fixer.training.sft --config ...` | SFT and checkpoint recovery |
| M4 | `python -m python_fixer.training.grpo --config ...` | Train-only reward adapter |
| M3/M4 | `python scripts/gpu_preflight.py --config ...` | Versioned stage-specific smoke report |

These entrypoints were planned in M0. M1 now implements `data.build` and
`sandbox.preflight`; see [implementation status](../docs/m1-implementation.md).
Training and model evaluation remain unimplemented. Entrypoints must fail on unresolved revision,
manifest, sandbox image or persistent storage requirements instead of using latest.

## Execution request v1

JSON object: `schema_version:1`, `request_id` (UUID), `run_id`, `problem_id`,
`candidate_sha256`, `candidate_source` (UTF-8, <=64 KiB), `suite_role:train_reward`
and `limits_profile:python-pure-v1`. Caller cannot provide test paths, commands,
Docker flags, dependency installation or a different role. Gateway recomputes the
hash and resolves a problem ID from its authenticated train-only allowlist.

Response: `schema_version`, `request_id`, `candidate_sha256`, `suite_sha256`,
`image_digest`, `policy_sha256`, `status`, `compiled`, counts `fixed_failed`,
`total_failed`, `preserved_passed`, `total_passed`, `all_passed`, `wall_ms`,
`cpu_ms`, `peak_memory_bytes`, `truncated_output`, `runner_version` and bounded
diagnostics. Status is one of OK, INVALID_OUTPUT, COMPILE_ERROR, POLICY_VIOLATION,
TIMEOUT, RESOURCE_LIMIT, INTERNAL_ERROR. The trusted runner derives all counts;
candidate stdout is never a result protocol. Partial result counts after crashes
are discarded. See [reward precedence](05-reward-engine.md).
The response shape is [sandbox-outcome.schema.json](../schemas/sandbox-outcome.schema.json);
M1 adds semantic/hash/count checks beyond JSON shape validation.

Infrastructure error: retry same immutable request at most twice with backoff,
then abort/checkpoint the training step. Do not turn unavailable infrastructure
into negative training rewards. Stable outcome caching keys include candidate,
suite, policy and image hashes. Disable caching for nondeterministic cases and
exclude them from the primary dataset.

## Trust and failure boundaries

GPU trainer has no container daemon permission. Gateway on the CPU worker controls
the runtime through a narrow authenticated interface. Its credentials are kept
outside candidate sandboxes. On timeout, destroy the entire container/cgroup,
not just its parent process. Limit global concurrency to four initially and queue
requests with a bounded capacity; backpressure pauses generation. A sandbox outage
blocks training, while unrelated CPU documentation/data metadata work may continue.

Use HTTPS with mutual TLS and an operator-issued CA, on a private network when
available. Map the client certificate identity to an immutable train-only study
allowlist; request fields cannot select permissions. The independent evaluator
uses a separate worker/test store and certificate. Gateway verifies request size,
schema and digest before enqueueing; audit only IDs/hashes/status, never secret
certificate material or private expected values. No anonymous public endpoint.

Checkpoints distinguish RUNNING, CHECKPOINTED, INTERRUPTED, FAILED, COMPLETED and
EVALUATED. Only artifact verification plus expected steps and independent evaluation
can promote a completed run to research evidence. No checkpoint contains held-out
tests or credentials. M1 must verify the interface before GPU integration.

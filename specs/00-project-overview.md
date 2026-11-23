# 00 — Project scope and research contract

## Outcome

Determine whether verified-pair SFT and GRPO improve Python function repair by
Qwen3-0.6B under controlled evaluation. Deliver code, versioned datasets, real
training artifacts, statistically qualified results and a reproducible report
for ML engineering / LLM post-training roles. Neither a chatbot nor prompt-only
demo meets this objective. Negative or inconclusive results remain valid outcomes.

## Scope

Initial tasks are self-contained deterministic Python functions with an explicit
signature, behavioral description and standard-library-only dependencies. Bugs
include boundary conditions, wrong comparisons/operators, accumulator/init errors,
incorrect branches and return values. Exclude interactive programs, networking,
filesystem tasks, concurrency, dependency repair and arbitrary repository patches.
Keep compatible function signatures; return one complete module, not a diff.
The harness owns invocation and tests. This constrained scope enables trustworthy
rewards and a tractable 24 GB GPU study; do not generalize results to repository
repair or unrestricted Python execution.

Planning sizes: 3,000–5,000 verified SFT pairs, 1,000–5,000 RL training prompts,
300–500 held-out problems, plus 200–500 validation problems. Sizes depend on the
number of distinct source families surviving verification; never multiply test
mutations to manufacture independent evaluation size. Freeze final counts and
attrition reasons before training. SFT and RL training sets may overlap within
the training partition, with exact overlap and sensitivity analyses reported.

## Hypotheses and arms

E0 evaluates the fixed base. E1 applies two planned epochs of LoRA SFT. E2 starts
from the matching E1 seed checkpoint and applies GRPO. E3 applies identical GRPO
to the base. E0-P uses a stronger fixed prompt with no weight updates. E2 versus
E1 isolates incremental RL; E2 versus E3 tests warm-up under matched RL updates,
candidate budget and data order, while including SFT cost in total-cost reports.
No hypothesis presumes gains. Exact model/tokenizer SHAs must be identical across
arms, other than trained adapters; 1.7B is an optional separate study.

## Delivery and evidence

M0 delivers these specifications and CPU checks. M1 delivers dataset and sandbox
verification. M2–M5 deliver actual baseline and training runs. M6 seals independent
evaluation and M7 publishes reproducible research artifacts. Details and dependencies
are in [the roadmap](../docs/implementation-roadmap.md).

Completion of research requires all four primary arms, three training seeds per
trained arm if affordable, fixed evaluation, artifact integrity and candid limits.
If budget permits only one seed, publish an explicitly exploratory pilot and defer
confirmatory claims. A process exit, low training loss or a high train reward is
insufficient. Never report unrun experiments as zero performance or PASS.

## Resource and authority assumptions

One rented Linux RTX 4090 24 GB node for training; separate CPU sandbox worker;
persistent artifact storage independent of node lifetime. Provider, hourly rate,
storage URI, GPU availability and spending cap are unknown. They block paid jobs,
not M1 CPU implementation. No automatic provisioning, weights download or training.
Versioned source is GitHub; datasets/checkpoints and private evaluator storage are
external. Credentials are securely injected at runtime, never committed.

## Requirements owned outside a training run

Each release must satisfy [requirements.json](../verification/requirements.json).
Task contracts cannot weaken data isolation, security tests or required evidence.
All GPU memory, throughput, training duration and cost estimates require preflight.
Scientific protocol changes after seeing held-out results require a new study ID
and fresh holdout; keep the original outcome in the report.

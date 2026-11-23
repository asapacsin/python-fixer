# 04 — GRPO / RLVR implementation contract

Use TRL **0.23.1** `GRPOTrainer`, `GRPOConfig`, PEFT and a callable reward adapter.
This is GRPO with explicit Dr. GRPO loss normalization, not an implementation of
the algorithm from scratch. The inspected version defaults to DAPO; every loss
setting below is explicit. Do not adopt a newer API without a versioned decision,
compatibility tests and updated experiment identity.

## Initialization and trainer API

E2 loads the selected E1 adapter into the same pinned BF16 base and resumes its
trainable LoRA parameters with a **new** RL optimizer/scheduler. It is not SFT
optimizer continuation. E3 creates a fresh LoRA adapter with the same rank/targets
on the base. Avoid double wrapping an already PEFT-wrapped model. `GRPOTrainer`
receives `model`, `processing_class=tokenizer`, `args=GRPOConfig(...)`, a training
dataset and `reward_funcs=[behavior_reward]`. Dataset rows contain `prompt` plus
opaque `problem_id`; `remove_unused_columns=False` preserves reward metadata.

Render string prompts using exactly the SFT/evaluation template and Qwen3
`enable_thinking=False`; do not feed conversational rows through a default
thinking-enabled chat-template path. Assert the prompt token prefix matches E0/E1.
`behavior_reward(prompts, completions, problem_id, **kwargs)` returns one finite
float per generated completion, in trainer order. Test metadata expansion and
group alignment with repeated problem IDs. No caller-provided tests or shell
execution. Reward callback may batch remote requests but must preserve order.

## Batches and sampling

[grpo.json](../configs/grpo.json): one device, microbatch **1 completion**,
accumulation 16, generation batch 16 completion sequences, `num_generations=4`.
Thus 4 distinct prompts × 4 candidates per generation/update window, **not** 16
prompts × 4 candidates. TRL's repeat sampler supplies grouped replicas; do not
manually duplicate the dataset. `generation_batch_size=16` implies
`steps_per_generation=16` on this device; do not set both fields. Generation batch
must be divisible by G and by device microbatch × world size. For G=8, the same
16-completion budget gives two prompts/group window; this ablation changes prompt
diversity and must be reported. Keep evaluation outside Trainer initially to avoid
its separate eval-batch divisibility requirements.

Temperature .7, top_p .8, top_k 20; max prompt 1536 and completion 512 tokens,
one response per sampler replica, EOS from the pinned tokenizer. Filter source
prompts exceeding 1536 before dataset freeze; never left-truncate away behavior.
Primary LR=5e-6, 10 warmup steps then constant LR, max_steps=500, one update reuse
(`num_iterations=1`), gradient checkpointing, BF16, no vLLM. These are unverified
starting values, not optimized claims. Log requested and observed counts to
catch an incorrect effective batch calculation.

## Objective and relative advantages

For prompt q and candidates i=1..G with scalar rewards r_i, let
`mu=mean(r)` and `s=sqrt(sum((r_i-mu)^2)/(G-1))` (sample standard deviation).
With `scale_rewards="group"`, TRL computes `A_i=(r_i-mu)/(s+1e-4)` and broadcasts
the sequence advantage over valid completion tokens. It logs near-zero std groups.
For policy ratio `rho_it=exp(log pi_theta(y_it|q,y_<t)-log pi_old(...))`, minimize

`L = -1/(B*Lmax) sum_i sum_t mask_it [min(rho_it*A_i,
clip(rho_it,0.8,1.2)*A_i) - beta*KL_it]`.

This states single-device Dr. GRPO normalization; TRL handles accumulation scaling.
`importance_sampling_level="token"`, `epsilon=.2`, `epsilon_high=.2`,
`loss_type="dr_grpo"`, `num_iterations=1`. For one on-policy update the old log
probabilities may be current detached log probabilities in this version; clipping
can be inactive initially. Do not imply PPO-style multiple updates occurred.
Masks exclude padding; `mask_truncated_completions=True` excludes unfinished
responses from loss, with truncation metrics still counted as failed attempts.

Primary `beta=0.0`: no reference model, reducing memory. Optional beta=.01
ablation freezes the **exact arm initialization**, including E1 adapter for E2.
With KL enabled, verify TRL PEFT reference handling: disabling adapters would
compare E2 to the base, not its SFT initializer. Use a separately frozen copy or
a supported named reference adapter, prove unchanged weights and correct logits
in smoke tests, and budget memory. KL estimator in this version is
`exp(log pi_ref-log pi_theta)-(log pi_ref-log pi_theta)-1` per valid token.

## Degenerate rewards and failures

All fail or all pass with equal rewards => A=0; do not synthesize positive labels,
resample until success or drop hard prompts silently. With beta=0 and no other
nonzero groups, the policy gradient contribution is zero. With beta>0 KL can
still update. Mixed failure statuses can have different penalties; otherwise equal
partial rewards are also zero-variance. Log fraction of zero-std, all-pass and
all-fail groups separately. No adaptive difficulty filtering in the primary arm.

Compile errors, invalid output, timeouts and disallowed operations follow
[reward v1](05-reward-engine.md). A regression yields zero behavior reward even
if some formerly failing tests are repaired. Sandbox INTERNAL_ERROR is not a
candidate failure: retry boundedly then checkpoint and mark BLOCKED.

## Stability, stopping and evidence

Record reward histograms, success rate, group std, advantages, truncation, lengths,
KL (when enabled), clip ratios, finite gradients, effective prompts/completions,
entropy if available, throughput and sandbox latency. Abort immediately on NaN,
source/test isolation violation or corrupt checkpoint. Pause for diagnosis if
truncation >20% or >90% zero-std groups over 100 consecutive groups. This is a
preregistered operational stop, not evidence of learning. Resume only under a
new versioned config if policy settings change. Do not preferentially stop E3
because it looks worse; report early-stop budget and reason.

M4 preflight: two prompts × four candidates then 2–5 real optimizer steps, verify
reward order/count, nonzero finite gradients for a mixed group, zero advantage
for equal groups, policy weight change and full checkpoint/reload. Save every
25 steps with RNG, optimizer, dataset/sampler position and reward version.
Select highest validation pass@1 at 25-step checkpoints, earliest tie. Never use
test rewards or held-out results for training or checkpoint selection.

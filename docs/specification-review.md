# M0 consistency review

Scope: author self-check against the pasted research brief and the reviewed
AI-native engineering source. **Not an independent reviewer verdict.**

## Resolved design ambiguities

- Qwen3 thinking is explicit false in SFT/RL/evaluation; prompts and EOS masks
  must be tested against the pinned tokenizer rather than assumed.
- TRL 0.23.1 defaults to DAPO. This study explicitly selects Dr. GRPO loss,
  sample-standard-deviation group scaling and token importance ratios.
- A 16-completion generation batch with G4 contains four prompts, not 16;
  microbatch/accumulation/divisibility rules are recorded and validated.
- Primary KL is disabled. A future nonzero-KL E2 ablation must freeze its SFT
  initializer; simply disabling PEFT adapters would use the wrong reference.
- Identical rewards yield zero relative advantages; no resampling-until-success
  or silent removal of difficult prompts is allowed.
- Positive behavior reward requires preserving baseline passing tests. Compile
  success alone earns no bonus; infrastructure failure is not a training penalty.
- Private expected test values stay outside candidate processes. Read-only
  pytest files inside a hostile interpreter are not sufficient test integrity.
- M1 preliminary data manifests are token-filtered in M2 before final freeze;
  identical eligibility and family isolation apply to every experimental arm.
- E2/E3 match RL budgets; extra SFT compute remains visible in total-cost analysis.
- Greedy primary and stochastic secondary pass@1 are separate metrics; paired
  family uncertainty and seed variability are not conflated with more problems.
- Test/model revisions and persistent artifact destinations remain null/unknown
  until real acquisition/preflight; no fabricated pins or completed runs.

## Remaining implementation and external gates

M1 must implement real dataset builders, production reward and gVisor gateway;
none exist in M0. M2 must implement sealed evaluation and metric statistics.
M3/M4 must resolve/hash-lock the proposed training stack, verify GPU/PEFT/TRL
compatibility and checkpoint recovery, with explicit spending authorization.
The provider, price, persistent storage, source licenses and model/dataset
commits are not yet selected/verified. Corpus counts and power remain estimates.

The local static checks and unit tests cover specification consistency only.
Hosted CI, branch rules, authenticated independent model review, model improvement
and secure runtime execution are NOT_VERIFIED or NOT_RUN as applicable. No
empty implementation directories or fake training CLI entrypoints were added.

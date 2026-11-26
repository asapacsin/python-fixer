# Seed 17 GRPO recovery strategy

Saved locally on 2026-11-26 (Asia/Shanghai).

## Verified findings

The seed 17 SFT adapter loads and generates on the cheaper inference node. The train-only diagnostic sampled four prompts with four candidates per prompt, using the same prompts and seed for two sampling settings.

| Setting | Temperature | top_p | top_k | Groups with unequal rewards |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.7 | 0.8 | 20 | 0/4 |
| Wider sampling | 1.2 | 0.99 | 0 | 3/4 |

Under the original settings, sampling filters left one eligible token at all 976 measured non-EOS token positions. Each group therefore generated four identical repairs. Wider sampling produced 10 distinct repairs summed across the four groups, compared with four in the original arm.

The isolated CPU scoring pilot used a 60-second wall limit and a 30-second CPU limit. All four train rows were reverified with three reference/buggy repeats under that policy. All 32 candidates completed scoring with zero timeouts. Wider sampling produced these rewards:

| Train problem | Rewards |
| --- | --- |
| max_profit | 1, 0, 1, 1 |
| is_happy | 1, 1, 1, 0 |
| lis_length | 0, 1, 1, 1 |
| peak_index | 0, 0, 0, 0 |

This establishes a reward signal on three sampled prompts. Wider sampling also reduced full-pass frequency in this sample. Improvement after GRPO remains unmeasured. Completed optimizer updates: zero.

## Execution plan

1. Use the existing seed 17 SFT adapter and pinned Qwen3-0.6B base. Start a fresh GRPO optimizer.
2. Use `configs/grpo-seed17-wide-pilot.json` for the two-update pilot. It specifies temperature 1.2, top_p 0.99, top_k 0, four candidates per prompt, 16 completions per update window, and a 128-token completion cap.
3. Score candidates on the isolated CPU worker with the verified 60/30 policy. Set the reward client's `request_timeout=90` so its transport deadline covers the worker execution window.
4. On the training GPU, verify finite gradients, an actual adapter parameter change from mixed rewards, and checkpoint save/reload during the two-update pilot. Measure generation, scoring and update times separately.
5. Before expanding to the full train partition, reverify the intended training rows under the extended policy and publish a fresh pinned manifest. The current pilot worker contains four selected rows from the 391-row train partition.
6. Use measured pilot throughput to estimate a longer run. Record the new sampling and execution-policy settings as a versioned experimental configuration. Preserve train split membership and report zero-variance groups and truncation. Stop for diagnosis if the registered limits in spec 04 are reached.

## Connection and scoring setup

The worker runs in the local isolated QEMU CPU guest. Its existing admin account is `ubuntu`; the existing guest SSH key works with that account. Private keys remain in their existing protected locations.

Avast intercepted TLS forwarded through Windows SSH and replaced the server certificate. Use SSH forwarding from WSL so the client can verify the worker's real certificate with the existing CA and client certificate.

The pilot service is `python-fixer-grpo-pilot`, listening at guest loopback `127.0.0.1:8444`. The WSL local forward is `127.0.0.1:18444` to that guest port. For a scoring session, keep a WSL reverse SSH tunnel open from inference/training-node loopback `127.0.0.1:41239` to WSL `127.0.0.1:18444`. The RewardClient endpoint is `https://localhost:41239` with certificate verification enabled. Recheck these listeners when resuming because tunnel processes and VM runtime state may expire.

The primary worker still uses 5 seconds wall time and 2 CPU seconds. Choose the extended pilot endpoint explicitly. The 60/30 limits remain bounded so a nonterminating candidate eventually exits.

## Artifacts and identity

- Measured results: `reports/evaluations/seed17-grpo-reward-pilot.json`.
- Two-update configuration: `configs/grpo-seed17-wide-pilot.json`.
- Reverification CLI: `scripts/reverify_train_subset.py`.
- Local raw candidate archive: `C:/Users/Administrator/Documents/Codex/2026-11-25/d/work/grpo-spread-recovery.json`.
- Local exported four-row registry: `C:/Users/Administrator/Documents/Codex/2026-11-25/d/work/grpo-pilot-extended` and its `.tar.gz` archive.
- Base model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Seed 17 adapter SHA-256: `a9e7892efa7f3a74a4b32eee9375beed66ced49d9af9e8f314c7ca7438710b90`.
- Source train JSONL SHA-256: `3cc3007c724d89692b24573af28f3882aba9407edabc4d07734ad8dfbcc67928`.
- Pilot canonical manifest SHA-256: `1093c8c4dd7c2047d7ca971cee8ae25c21b724799d9e5c419f0ee46181a37ab9`.
- Extended worker policy SHA-256: `ef57d86418574c34a58413d6986cc9d0b5e81aa08d4d0fe6a1f4ffddcec4c435`.

Resume from these local artifacts and verify their hashes. The remaining training checks and full-partition reverification are pending.

## Two-step optimizer smoke

Completed on 2026-11-26 on the inference GPU with the four-row verified pilot worker. The run completed 2 optimizer steps and used the hard budget of 32 candidates. All rewards were finite. Both steps had finite nonzero gradient norms (0.148 and 0.115). 392 adapter tensors changed. The step-2 checkpoint manifest verified, and reloading the final adapter reproduced the trained adapter tensor hashes.

The run logged 8 reward groups, of which 3 had zero variance. The remaining 5 groups had mixed rewards. This verifies the optimizer and checkpoint path for the pilot subset. It does not estimate quality or validate all 391 training rows. The full 391-row train partition is being reverified under the extended policy on the isolated CPU guest. It is estimated to take 2–4 hours. Full training remains unstarted.

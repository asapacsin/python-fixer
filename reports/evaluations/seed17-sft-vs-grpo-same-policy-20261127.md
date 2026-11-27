# Seed 17 SFT vs GRPO: same-policy comparison

Run date: 2026-11-27 (Asia/Shanghai). Both adapters were evaluated greedily on the same 32 validation problems with the same 15 s / 5 CPU gVisor policy and the same generation script.

| Model | Solved | Pass@1 |
| --- | ---: | ---: |
| SFT seed 17 | 18/32 | 56.25% |
| GRPO seed 17 | 19/32 | 59.38% |

GRPO gained **1 problem out of 32 (+3.13 percentage points)** over SFT: 1 improved, 0 regressed. GRPO was initialized from this exact SFT adapter (SHA-256 `a9e7892efa7f3a74a4b32eee9375beed66ced49d9af9e8f314c7ca7438710b90`).

This is an exploratory single-seed comparison. GRPO training used four verified train-only rows and 30 optimizer steps; it is not the full 391-row run. The result does not support describing the GRPO gain as a large improvement.

The earlier **2/32** figure was the unadapted base model under this same scoring policy; it was not the SFT score. The earlier **10/32 SFT** report used a different, stricter 5 s / 2 CPU policy and is not directly comparable to this result.

Full paired per-problem results, candidate hashes, policy hash and image digest are in the companion JSON. The GRPO adapter remains at `outputs/grpo-seed17-one-hour-20261127/final-adapter/`.


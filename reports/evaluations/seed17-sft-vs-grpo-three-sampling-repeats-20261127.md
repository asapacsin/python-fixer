# SFT seed 17 vs GRPO seed 17: three paired sampling repeats

The same fixed SFT and GRPO adapters were evaluated on the same 32 validation problems with three matched sampling seeds. Both models used temperature 0.7, top-p 0.95, top-k 0, a 512-token cap, and the same gVisor policy (15 seconds / 5 CPUs).

| Sampling seed | SFT solved | GRPO solved | Difference | Improved | Regressed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 16/32 (50.00%) | 17/32 (53.13%) | +1 | 1 | 0 |
| 29 | 17/32 (53.13%) | 17/32 (53.13%) | 0 | 0 | 0 |
| 43 | 15/32 (46.88%) | 18/32 (56.25%) | +3 | 3 | 0 |
| Mean | 16.00/32 (50.00%) | 17.33/32 (54.17%) | **+1.33/32 (+4.17 pp)** | 4 paired wins | 0 paired losses |

Across the 32 problem clusters, GRPO was better on 4, worse on 0, and tied on 28. The exact paired sign test gives one-sided *p* = 0.0625 (two-sided *p* = 0.125). This is a small positive trend, but it does **not** establish a reliable gain at the conventional 0.05 level.

These are three stochastic inference repeats of the **same fixed checkpoints**, not three independent training seeds. They estimate decoding variability only. The GRPO adapter was initialized from SFT seed 17 and trained on four verified train-only rows. Results remain exploratory for the 32-row validation set.

The full per-problem results and hashes are in the companion JSON. The original deterministic comparison was SFT 18/32 vs GRPO 19/32 under the same policy; the repeated sampled evaluation is reported separately because it uses stochastic decoding.

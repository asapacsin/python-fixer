# Qwen3-0.6B SFT validation comparison

**Observed result:** SFT seed 17 solved **10/32 (31.3%)** validation problems; the base model solved **1/32 (3.1%)**. The paired change is **+9/32, or +28.1 percentage points**. SFT fixed 9 baseline failures and regressed 0 baseline passes.

| Arm | Solved | Greedy pass@1 |
| --- | ---: | ---: |
| Qwen3-0.6B baseline | 1 / 32 | 3.1% |
| Qwen3-0.6B + SFT seed 17 | 10 / 32 | 31.3% |
| Paired difference | +9 / 32 | +28.1 pp |

Both arms used the same pinned base revision, tokenizer, prompt construction, BF16 H200 inference, non-thinking greedy decode, batch size 1, and 512-token cap. Generated candidates were scored in the separate QEMU CPU worker using the pinned gVisor image and sandbox policy; the worker isolation preflight passed 35/35 checks. No test-split rows were used.

This is an **exploratory validation result**, not the final held-out test result. It describes one 32-problem validation set and one SFT seed; it does not establish a confirmatory improvement or statistical significance. Per-problem results and provenance hashes are in the JSON artifact. Raw candidate completions were kept outside the repository.
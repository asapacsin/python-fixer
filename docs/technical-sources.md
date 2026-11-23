# Technical source verification

Checked 2026-11-23. [source-checks.json](source-checks.json) records source URLs,
retrieved byte hashes and successful retrieval sizes. This is a documentation
review, not a package compatibility, hardware or experiment result. A model card
at `main` is mutable; implementation must resolve immutable Hub revisions.

| Primary source | Confirmed detail | Project decision |
|---|---|---|
| [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/README.md) | Transformers <4.51.0 lacks Qwen3 support; thinking on by default; explicit false supported; non-thinking recommendation T=.7, p=.8, k=20 | Explicit non-thinking across train/eval; primary greedy deliberately separated from recommended sampled secondary |
| [TRL 0.23.1 GRPOConfig](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_config.py) | G>=2; generation-batch divisibility; default DAPO loss; group/none/batch scaling; beta=0 avoids reference model | Pin API, set dr_grpo, G4 and generation batch16 explicitly |
| [TRL 0.23.1 GRPOTrainer](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py) | Group sample std, +1e-4 denominator; token ratio/clipping; constant max-length Dr. GRPO normalization | State exact objective and zero-variance behavior; test real batch metadata |
| [TRL 0.23.1 SFTConfig](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/sft_config.py) and [SFTTrainer](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/sft_trainer.py) | max_length, prompt-completion loss masking and assistant-only mode are separate controls | Pre-rendered prompt/completion, explicit completion-only mask, packing off |
| [QuixBugs README](https://github.com/jkoppel/QuixBugs) | Small multi-language algorithm benchmark, not thousands of independent Python functions | Optional diagnostic; per-file licensing and public contamination audit |
| [BugsInPy README](https://github.com/soarsmu/BugsInPy) | Reproduction-oriented real Python project bugs with environments | Separate future external-validity track; not drop-in primary function corpus |
| [gVisor Docker quick start](https://gvisor.dev/docs/user_guide/quick_start/docker/) | runsc runtime integration exists | Require separate CPU worker and actual isolation tests; docs do not certify local support |

Reference documentation to verify at GPU implementation: [PyTorch previous
versions](https://pytorch.org/get-started/previous-versions/) for exact 2.8.0
cu128 wheel/index; [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
for provider driver compatibility; [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora)
for quantized adapter preparation. These pages are implementation prerequisites,
not claimed as installed/validated in M0.

Method references for implementation review: DeepSeekMath
([arXiv:2402.03300](https://arxiv.org/abs/2402.03300), GRPO), Understanding R1-Zero-Like
Training ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783), Dr. GRPO), and
Evaluating Large Language Models Trained on Code
([arXiv:2107.03374](https://arxiv.org/abs/2107.03374), pass@k estimator).
These references motivate choices; no paper's reported performance is imported
as an expectation for this study. The current specification's exact implementation
semantics are grounded in the versioned TRL sources above.

## Open checks

Full dependency resolver/wheel hashes, model and tokenizer commits, actual
Qwen3/PEFT/TRL GPU execution, runsc availability and kernel behavior, all source
licenses, dataset sizes, clone clusters, test quality, GPU memory/throughput,
persistence recovery, provider price and independent reviewer remain unverified.
These are milestone gates, not optional assumptions to bypass.

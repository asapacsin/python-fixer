# 03 — Supervised fine-tuning

## Model and formatting

Pin Qwen/Qwen3-0.6B model and tokenizer to the same immutable Hub commit, record
tokenizer files and chat template hashes. Qwen3 requires Transformers >=4.51.0;
the proposed stack uses 4.56.2 with TRL 0.23.1. These APIs were inspected, not
GPU-certified; see [sources](../docs/technical-sources.md).

Use `enable_thinking=False` explicitly. Default Qwen3 thinking mode is unsuitable
for the primary code-only output contract. Do not train `<think>` traces. The fixed
user template is [repair.txt](../configs/prompts/repair.txt). No reference, tests,
reward summary or provenance in prompts. Target: one complete Python module with
the original named function/signature, optional permitted helper functions, raw
code only, no fences or prose. Format violations are scored invalid consistently.

Serialize a non-conversational prompt/completion SFT dataset after applying the
tokenizer chat template to system/user messages with `add_generation_prompt=True`
and `enable_thinking=False`. Produce completion bytes from reference source plus
one turn-ending `<|im_end|>` token. Do not append EOS twice or reapply templates.
Use `SFTTrainer(processing_class=tokenizer, peft_config=LoraConfig(...),
args=SFTConfig(...), train_dataset=...)` and explicit `completion_only_loss=True`,
`assistant_only_loss=False`, `packing=False`. Prompt/completion masks avoid relying
on a template's optional assistant-generation mask. Test token-prefix equality,
nonempty target, EOS handling and decoded examples. All prompt/pad tokens have
labels -100, code and one EOS are trainable; assert this on actual collated batches.
Set pad token from tokenizer if provided, otherwise EOS with explicit attention
masks; never infer attention from token equality alone.

## Initial 24 GB configuration

[sft.json](../configs/sft.json) is the authoritative proposed configuration:
BF16 LoRA r=16, alpha=32, dropout=.05 on attention and MLP projection modules,
AdamW lr=2e-4, cosine schedule, warmup 5%, weight decay .01, grad norm 1,
two epochs, microbatch 1, accumulation 16 => 16 examples/update on one GPU.
Maximum prompt+completion 2048 tokens. Exclude/quarantine examples that do not fit
after templating; do not silently truncate the corrected program or behavioral
description. Use gradient checkpointing, `use_cache=False` during training and
restore cache for inference. Prefer native SDPA first; no FlashAttention/vLLM
dependency in the initial smoke test.

BF16 LoRA is the primary path for 0.6B, avoiding quantization confounds. Estimated
base weights ~1.2 GB at two bytes per parameter; activations, temporary logits,
optimizer, gradients and generation dominate additional memory and are unmeasured.
Reserve >=2 GB device headroom; pass preflight only if allocated peak <22 GiB on
the target 24 GB device and no OOM. This is a safety threshold, not a memory claim.
If OOM: reduce sequence length through a *new dataset eligibility manifest* or
microbatch first; QLoRA is an explicitly labeled fallback/ablation using NF4,
double quantization, BF16 compute and `prepare_model_for_kbit_training` before
PEFT wrapping. Do not compare a quantized E2 against unquantized E3 inadvertently.

## Training evidence and checkpointing

M3 preflight: load actual pinned model/tokenizer, inspect trainable parameter
names/counts, verify only LoRA parameters require gradients, overfit 16 verified
training examples for 20–50 steps. Assert finite loss/gradients, nonzero adapter
gradient, changed adapter tensor hashes and decreasing toy-set loss. Preserve
before/after decoded repairs; a change in output is not proof of better heldout
performance. Confirm frozen base tensor hashes unchanged on sampled layers.
Run validation inference, not just teacher-forced loss. Test adapter unload/reload
reproduces logits within recorded BF16 tolerance (start atol=.02, rtol=.02 and
document measured deviations); greedy repair outputs should match exactly.

Log step, loss, validation loss, grad norm, learning rate, token counts,
examples/sec, device peak allocated/reserved memory, wall time and cost estimate.
Checkpoint every 50 steps and at termination, including adapter, optimizer,
scheduler, RNG states, trainer state, sampler position and run manifest. Persist
to the external run root; the adapter alone is an export, not a resumable run.
Use latest *verified* checkpoint for recovery, never a partially written directory.

## Selection and reproducibility

Run seeds 17/29/43 with identical frozen train/validation family splits. Evaluate
validation repair pass@1 every 50 steps. Select highest validation pass@1,
tie-break earliest step, never held-out score. Record full trajectory and compute.
Fix dataloader seeds and deterministic operations where supported; document
nondeterministic CUDA kernels. Interrupted/resumed runs must record deviations
from uninterrupted runs. Optional LoRA rank/QLoRA tuning consumes only a fixed
validation budget and is locked before confirmatory test evaluation.

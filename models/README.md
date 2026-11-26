# Qwen3-0.6B SFT E1 adapters

These are LoRA adapter weights, not full copies of the base model. Load with the pinned base model Qwen/Qwen3-0.6B at revision c1899de289a04d12100db370d81485cdf75e47ca.

Each seed directory contains adapter_model.safetensors and adapter_config.json. The run manifests, training metrics, evaluation details, and SHA-256 hashes are in reports/runs/ and reports/evaluations/.

To use an adapter, load the pinned base model and attach the corresponding directory with PEFT. Evaluation results are exploratory and documented in reports/evaluations/python-fixer-E1-three-seed-validation-20261126.md.

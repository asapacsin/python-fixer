"""Tiny, locally constructed Qwen3-architecture model and ChatML tokenizer.

Random weights, no download. These exercise the library wiring only; they say
nothing about the real pinned Qwen3 tokenizer or model quality.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirrors the Qwen3 non-thinking generation prompt shape (empty think block).
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}")


def training_available():
    try:
        import torch, transformers, trl, peft  # noqa: F401
        return True
    except ImportError:
        return False


def train_rows():
    path = ROOT / "verification/evidence/m1-v7/m1-fixtures-v7/train.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [r["buggy_code"] + r["reference_code"] + r["behavior"] for r in train_rows()]
    corpus.append((ROOT / "configs/prompts/repair.txt").read_text())
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=specials[1:],
                                   model_input_names=["input_ids", "attention_mask"])
    fast.chat_template = CHAT_TEMPLATE
    return fast


def tiny_model(tokenizer, seed=0):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
                      pad_token_id=tokenizer.pad_token_id, use_cache=False)
    return Qwen3ForCausalLM(cfg)


def cpu_config(cfg):
    """Spec config with only device-specific fields relaxed for a CPU test."""
    out = json.loads(json.dumps(cfg))
    out["trainer"].update(bf16=False, gradient_checkpointing=False, gradient_accumulation_steps=1,
                          eval_strategy="no", logging_steps=1, report_to="none")
    return out

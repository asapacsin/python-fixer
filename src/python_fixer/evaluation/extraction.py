"""Strict, text-only extraction of a candidate module from a raw model completion.

Spec 05 (output and policy checks) and spec 03 (code-only output contract):
exactly one UTF-8 module, <=64 KiB, no Markdown fences or thinking traces.
This module never parses, compiles or executes candidate code; AST/signature
policy is enforced inside the approved worker boundary. Format violations are
returned as INVALID results (not exceptions) so every arm scores them the same;
only a non-str/bytes argument (a caller bug) raises TypeError. Prose outside a
fence is not detected here; it surfaces as a worker COMPILE_ERROR.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

MAX_MODULE_BYTES = 64 * 1024
TURN_END = "<|im_end|>"
# Fences are banned anywhere, even inside string literals or comments: the output
# contract is raw code, and a fence is far more often Markdown than data.
_FORBIDDEN_MARKERS = ("```", "~~~", "<think>", "</think>", "<|im_start|>", "<|endoftext|>",
                      "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
                      "<|fim_suffix|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>")


@dataclass(frozen=True)
class Extraction:
    status: str  # "VALID" or "INVALID"
    reason: str | None
    module: bytes | None
    sha256: str | None


def _invalid(reason: str) -> Extraction:
    return Extraction("INVALID", reason, None, None)


def extract_module(completion: str | bytes) -> Extraction:
    if isinstance(completion, bytes):
        try:
            text = completion.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return _invalid("not_utf8")
    elif isinstance(completion, str):
        text = completion
    else:
        raise TypeError("completion must be str or bytes")
    # Allow at most one trailing turn-ending token, as produced by generation.
    if text.endswith(TURN_END):
        text = text[: -len(TURN_END)]
    if TURN_END in text:
        return _invalid("extra_turn_token")
    if "\x00" in text:
        return _invalid("nul_byte")
    if "�" in text:
        return _invalid("replacement_character")
    for marker in _FORBIDDEN_MARKERS:
        if marker in text:
            return _invalid("forbidden_marker")
    if not text.strip():
        return _invalid("empty")
    try:
        module = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return _invalid("not_utf8")
    if len(module) > MAX_MODULE_BYTES:
        return _invalid("too_large")
    return Extraction("VALID", None, module, hashlib.sha256(module).hexdigest())

"""Executable reward-v1 specification over trusted counts; NOT a sandbox or trainer.

M1 must implement the production reward independently and use the shared fixture
table as acceptance input. This module never parses or executes candidate code.
"""
from __future__ import annotations


class InvalidOutcome(ValueError):
    """No valid reward exists; retry infrastructure or reject the report."""


def specification_reward(outcome: dict) -> float:
    status = outcome.get("status")
    if status == "INTERNAL_ERROR":
        raise InvalidOutcome("infrastructure errors must not become rewards")
    if status in {"POLICY_VIOLATION", "TIMEOUT", "RESOURCE_LIMIT"}:
        return -1.0
    if status in {"INVALID_OUTPUT", "COMPILE_ERROR"}:
        return -0.5
    if status != "OK" or outcome.get("compiled") is not True:
        raise InvalidOutcome("missing, unknown or contradictory execution status")
    keys = ("fixed_failed", "total_failed", "preserved_passed", "total_passed")
    if any(type(outcome.get(k)) is not int for k in keys):
        raise InvalidOutcome("counts must be integers, not booleans")
    f, total_f, p, total_p = (outcome[k] for k in keys)
    if total_f < 1 or total_p < 1 or not 0 <= f <= total_f or not 0 <= p <= total_p:
        raise InvalidOutcome("invalid baseline or candidate counts")
    all_passed = f == total_f and p == total_p
    if type(outcome.get("all_passed")) is not bool or outcome["all_passed"] != all_passed:
        raise InvalidOutcome("full-pass flag disagrees with trusted counts")
    if p < total_p:
        return 0.0
    return 0.5 * f / total_f + 0.5 * int(all_passed)

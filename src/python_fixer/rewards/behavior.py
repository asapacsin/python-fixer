"""Production behavior-v1 reward; counts must come from the trusted gateway."""
from __future__ import annotations


class InvalidOutcome(ValueError):
    """Infrastructure or contradictory evidence: no scalar reward is available."""


def behavior_reward(outcome: dict, *, expected_bindings: dict | None = None) -> float:
    if not isinstance(outcome, dict):
        raise InvalidOutcome("expected gateway outcome")
    if expected_bindings is not None:
        for key, expected in expected_bindings.items():
            if outcome.get(key) != expected:
                raise InvalidOutcome("gateway binding mismatch: " + key)
    status = outcome.get("status")
    penalties = {"POLICY_VIOLATION": -1.0, "TIMEOUT": -1.0,
                 "RESOURCE_LIMIT": -1.0, "INVALID_OUTPUT": -0.5, "COMPILE_ERROR": -0.5}
    if status in penalties:
        return penalties[status]
    if status != "OK" or outcome.get("compiled") is not True:
        raise InvalidOutcome("incomplete execution; do not reward infrastructure failure")
    counts = [outcome.get(k) for k in ("fixed_failed", "total_failed", "preserved_passed", "total_passed")]
    if any(type(c) is not int for c in counts):
        raise InvalidOutcome("counts must be integers")
    fixed, failing, preserved, passing = counts
    if not (failing > 0 and passing > 0 and 0 <= fixed <= failing and 0 <= preserved <= passing):
        raise InvalidOutcome("invalid baseline/counts")
    complete = fixed == failing and preserved == passing
    if type(outcome.get("all_passed")) is not bool or outcome["all_passed"] != complete:
        raise InvalidOutcome("all_passed disagrees with counts")
    if preserved != passing:
        return 0.0
    return fixed / (2 * failing) + (0.5 if complete else 0.0)

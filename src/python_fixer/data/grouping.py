"""Group trusted reference functions before mutation; never executes source."""
from __future__ import annotations
import ast
from difflib import SequenceMatcher
from python_fixer.common import digest, text_digest


def normalized_source(source: str) -> str:
    return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)


def clone_shape(source: str) -> str:
    tree = ast.parse(source)
    names = {}
    def rename(value):
        return names.setdefault(value, "v" + str(len(names)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = rename(node.name)
        elif isinstance(node, ast.Name):
            node.id = rename(node.id)
        elif isinstance(node, ast.arg):
            node.arg = rename(node.arg)
    # Literals/operators remain: renaming cannot defeat a clone check.
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def group_families(seeds: list[dict], threshold: float = 0.92) -> dict[str, str]:
    ordered = sorted(seeds, key=lambda s: s["source_function_id"])
    parent = {s["source_function_id"]: s["source_function_id"] for s in ordered}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    shapes = {s["source_function_id"]: clone_shape(s["reference_code"]) for s in ordered}
    for i, left in enumerate(ordered):
        for right in ordered[i+1:]:
            a, b = left["source_function_id"], right["source_function_id"]
            related = left["source_family_id"] == right["source_family_id"]
            similar = SequenceMatcher(None, shapes[a], shapes[b], autojunk=False).ratio() >= threshold
            if related or similar:
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
    members = {}
    for identifier in parent:
        members.setdefault(find(identifier), []).append(identifier)
    stable = {leader: "family-" + digest(sorted(group))[:16] for leader, group in members.items()}
    return {identifier: stable[find(identifier)] for identifier in parent}


def split_for_family(family: str, seed: int = 17) -> str:
    bucket = int(text_digest(f"split-v1:{seed}:{family}")[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"

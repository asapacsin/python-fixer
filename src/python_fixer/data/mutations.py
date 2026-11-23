"""Deterministic single-AST-edit mutations of trusted, licensed seed source.

Each variant changes exactly one operator token in the original source text, so
the buggy module keeps the author's formatting and the repair is a one-token diff.
(Earlier versions re-emitted the whole module with ast.unparse, which also
reformatted unrelated lines, e.g. `a, b = 0, n` -> `a, b = (0, n)`.) A variant is
kept only if the edited text parses to exactly the intended mutated AST.
Source is parsed, never compiled or executed.
"""
from __future__ import annotations
import ast
import copy

REPLACEMENTS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
                ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add}
TOKENS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
          ast.NotEq: "!=", ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}


def _offsets(source: str) -> list[int]:
    starts, total = [], 0
    for line in source.splitlines(keepends=True):
        starts.append(total)
        total += len(line.encode("utf-8"))
    return starts


def _splice(source: str, left: ast.AST, right: ast.AST, old: type, new: type) -> str | None:
    """Replace the operator token between two operand nodes; None if not unambiguous."""
    if left.end_lineno != right.lineno:
        return None
    raw = source.encode("utf-8")
    starts = _offsets(source)
    begin = starts[left.end_lineno - 1] + left.end_col_offset
    end = starts[right.lineno - 1] + right.col_offset
    gap = raw[begin:end].decode("utf-8")
    token = TOKENS[old]
    if gap.strip("() \t") != token:
        return None
    position = gap.index(token)
    edited = gap[:position] + TOKENS[new] + gap[position + len(token):]
    return (raw[:begin] + edited.encode("utf-8") + raw[end:]).decode("utf-8")


def mutate(source: str) -> list[tuple[str, str]]:
    original = ast.parse(source)
    variants = []
    for index, node in enumerate(ast.walk(original)):
        edits = []
        if isinstance(node, ast.Compare):
            for pos, op in enumerate(node.ops):
                if type(op) in REPLACEMENTS:
                    left = node.left if pos == 0 else node.comparators[pos - 1]
                    edits.append(("ops", pos, type(op), left, node.comparators[pos]))
        elif isinstance(node, ast.BinOp) and type(node.op) in REPLACEMENTS:
            edits.append(("op", None, type(node.op), node.left, node.right))
        for field, position, old, left, right in edits:
            new = REPLACEMENTS[old]
            tree = copy.deepcopy(original)
            target = list(ast.walk(tree))[index]
            if position is None:
                setattr(target, field, new())
            else:
                getattr(target, field)[position] = new()
            text = _splice(source, left, right, old, new)
            if text is None or ast.dump(ast.parse(text)) != ast.dump(tree):
                continue
            variants.append((f"ast-{index}-{field}-{position}-{new.__name__}", text))
    return variants

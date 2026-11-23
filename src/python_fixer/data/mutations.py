"""Deterministic single-AST-edit mutations of trusted, licensed seed source."""
from __future__ import annotations
import ast
import copy

REPLACEMENTS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
                ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add}


def mutate(source: str) -> list[tuple[str, str]]:
    original = ast.parse(source)
    variants = []
    for index, node in enumerate(ast.walk(original)):
        replacements = []
        if isinstance(node, ast.Compare):
            replacements = [("ops", pos, REPLACEMENTS[type(op)]) for pos, op in enumerate(node.ops)
                            if type(op) in REPLACEMENTS]
        elif isinstance(node, ast.BinOp) and type(node.op) in REPLACEMENTS:
            replacements = [("op", None, REPLACEMENTS[type(node.op)])]
        for field, position, replacement in replacements:
            tree = copy.deepcopy(original)
            target = list(ast.walk(tree))[index]
            if position is None:
                setattr(target, field, replacement())
            else:
                getattr(target, field)[position] = replacement()
            variants.append((f"ast-{index}-{field}-{position}-{replacement.__name__}", ast.unparse(tree) + "\n"))
    return variants

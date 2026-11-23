"""Canonical serialization and atomic artifact writing, without code execution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sysconfig


def resource_path(relative: str) -> Path:
    checkout = Path(__file__).resolve().parents[2] / relative
    if checkout.is_file():
        return checkout
    installed = Path(sysconfig.get_path("data")) / "share" / "python-fixer" / relative
    if not installed.is_file():
        raise FileNotFoundError("missing project resource: "+relative)
    return installed


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strict_loads(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("non-finite JSON value: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(path, canonical_bytes(value) + b"\n")


def write_new_json(path: Path,value) -> None:
    """Atomically publish a new immutable artifact; concurrent writers cannot overwrite."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            stream.write(canonical_bytes(value)+b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.link(temporary,path)  # Same-directory hardlink, fails if target already exists.
        finally:
            temporary.unlink(missing_ok=True)

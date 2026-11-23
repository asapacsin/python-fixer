"""Export only the private train registry and model-facing train views."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from python_fixer.common import atomic_write, canonical_bytes, digest, write_json
from .artifacts import load_verified_partition
from .models import model_view


def export_train(dataset: Path, destination: Path, expected_manifest_sha256: str):
    rows, manifest = load_verified_partition(dataset,"train",expected_manifest_sha256=expected_manifest_sha256)
    if destination.exists():
        raise ValueError("refuse to overwrite export")
    destination.mkdir(parents=True)
    atomic_write(destination/"train.jsonl",(dataset/"train.jsonl").read_bytes())
    for row in rows:
        name = row["problem_id"]+".json"
        atomic_write(destination/"evidence"/name,(dataset/"evidence"/name).read_bytes())
    for role in ("sft","rl"):
        payload = b"".join(canonical_bytes(model_view(row,role))+b"\n" for row in rows)
        atomic_write(destination/(role+"-view.jsonl"),payload)
    # Original manifest/pin is unchanged; held-out metadata is not a held-out test bundle.
    write_json(destination/"manifest.json",manifest)
    return {"train_rows":len(rows),"manifest_sha256":digest(manifest),"heldout_tests_exported":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--destination",required=True)
    parser.add_argument("--manifest-sha256",required=True)
    args=parser.parse_args(argv)
    try:
        result=export_train(Path(args.dataset),Path(args.destination),args.manifest_sha256)
        print(canonical_bytes(result).decode())
        return 0
    except Exception as exc:
        print("EXPORT BLOCKED:",str(exc),file=sys.stderr)
        return 3


if __name__=="__main__":
    raise SystemExit(main())

"""Validate/analyze frozen evaluator results. Generation is not implemented here."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
from python_fixer.common import digest, strict_loads, write_new_json
from .plan import validate_plan
from .records import jsonl_records, validate_records
from .analysis import analyze_validated


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["validate","analyze"])
    parser.add_argument("--plan",required=True)
    parser.add_argument("--plan-sha256",required=True,help="operator-pinned canonical JSON digest")
    parser.add_argument("--results",required=True)
    parser.add_argument("--evidence-root",help="required for observed-model inputs")
    parser.add_argument("--output",help="new JSON artifact; existing files are never overwritten")
    args=parser.parse_args(argv)
    try:
        output=Path(args.output) if args.output else None
        if output and output.exists():
            raise ValueError("analysis output exists; choose a new immutable artifact path")
        with Path(args.plan).open("rb") as stream:
            plan_bytes=stream.read(32*1024*1024+1)
        if len(plan_bytes)>32*1024*1024:
            raise ValueError("frozen plan exceeds 32 MiB bound")
        plan=strict_loads(plan_bytes.decode())
        if digest(plan)!=args.plan_sha256:
            raise ValueError("plan differs from operator-pinned hash")
        validate_plan(plan)
        results_path=Path(args.results)
        # Hash the same exact bytes that are validated; no separate pre-read race.
        file_hash=hashlib.sha256()
        rows=validate_records(plan,jsonl_records(results_path,hasher=file_hash),
                              evidence_root=Path(args.evidence_root) if args.evidence_root else None)
        report=analyze_validated(plan,rows) if args.action=="analyze" else {
            "status":"VALIDATED_COMPLETE_GRID","plan_sha256":digest(plan),"records":len(rows),
            "evidence_kind":plan["evidence_kind"],"models_executed":False}
        report["results_file_sha256"]=file_hash.hexdigest()
        if output:
            write_new_json(output,report)
        print(f"{report['status']}: {len(rows)} records; evidence_kind={plan['evidence_kind']}")
        if output:
            print("Artifact:",output)
        return 0
    except Exception as exc:
        print("EVALUATION BLOCKED:",str(exc),file=sys.stderr)
        return 3


if __name__=="__main__":
    raise SystemExit(main())

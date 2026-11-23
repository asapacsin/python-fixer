#!/usr/bin/env python3
"""Capture real CPU checks bound to exact source; never executes candidates."""
import argparse
import datetime as dt
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

from validate_repository import ROOT, report_path, source_snapshot
from python_fixer.common import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",default=".ai-native/cpu-readiness.json")
    args=parser.parse_args()
    destination=report_path(ROOT,args.report)
    before=source_snapshot(ROOT)
    gates=[]
    commands=[("repository-contracts",[sys.executable,"scripts/validate_repository.py"],30),
              ("cpu-tests",[sys.executable,"-m","unittest","discover","-s","tests","-v"],120),
              ("metadata-plan",[sys.executable,"-m","python_fixer.data.build","--config","configs/data.json","--plan-only"],30)]
    for name,command,timeout in commands:
        try:
            done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
            output=done.stdout+done.stderr
            status="PASS" if done.returncode==0 else "FAIL"
            record={"id":name,"command":command,"exit_code":done.returncode,"status":status,
                    "output_tail":output[-32000:]}
            if name=="cpu-tests":
                found=re.search(r"Ran (\d+) tests? in",output)
                count=int(found.group(1)) if found else 0
                skipped=re.search(r"skipped=(\d+)",output)
                record.update(tests=count,skipped=int(skipped.group(1)) if skipped else 0)
                if count==0:
                    record["status"]="FAIL"
            gates.append(record)
        except subprocess.TimeoutExpired:
            gates.append({"id":name,"command":command,"status":"FAIL","reason":"timeout"})
    after=source_snapshot(ROOT)
    stable=before["fingerprint"]==after["fingerprint"]
    passed=stable and all(g["status"]=="PASS" for g in gates)
    report={"schema_version":1,"scope":"CPU metadata, rewards, evidence and TLS checks only",
            "status":"PASS" if passed else "FAIL","timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "snapshot":after,"source_stable_during_checks":stable,"gates":gates,
            "packages":{name:version(name) for name in ("python-fixer","jsonschema","setuptools")},
            "sandbox_security_acceptance":"NOT_VERIFIED_BY_CPU_GATES",
            "models_gpu_training":"NOT_RUN","independent_code_review":"NOT_VERIFIED_BY_CPU_GATES"}
    write_json(destination,report)
    print(f"{report['status']}: {len(gates)} CPU gates; evidence {destination}")
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())

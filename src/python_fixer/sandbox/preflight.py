"""Exercise actual gVisor isolation. No PASS is emitted without runtime checks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from python_fixer.common import write_json
from .docker import DockerSandbox
from .policy import Policy


def run_preflight(sandbox: DockerSandbox) -> dict:
    sandbox.available()
    cases = [
        ("normal-return", "def repair(x):\n    return x + 1\n", "OK"),
        ("forbidden-import", "import os\ndef repair(x):\n    return x\n", "POLICY_VIOLATION"),
        ("forbidden-read", "def repair(x):\n    return open('/etc/shadow').read()\n", "POLICY_VIOLATION"),
        ("introspection", "def repair(x):\n    return x.__class__\n", "POLICY_VIOLATION"),
        ("signature-change", "def changed(x):\n    return x\n", "POLICY_VIOLATION"),
        ("syntax", "def repair(:\n", "COMPILE_ERROR"),
        ("loop", "def repair(x):\n    while True:\n        pass\n", "LIMIT"),
        ("memory", "def repair(x):\n    return [0] * (10 ** 9)\n", "RESOURCE_LIMIT"),
        ("forged-pass", "def repair(x):\n    return {'all_tests_pass': True}\n", "OK"),
    ]
    checks = []
    for name, code, expected in cases:
        result = sandbox.invoke(code, "repair(x)", {"args": [1], "kwargs": {}})
        success = result["status"] in {"TIMEOUT", "RESOURCE_LIMIT"} if expected == "LIMIT" else result["status"] == expected
        if name == "normal-return":
            success &= result.get("value") == 2
        if name == "forged-pass":
            success &= result.get("value") != 2
        checks.append({"id": name, "passed": success, "status": result["status"]})
    # Bypass the AST filter with trusted probes to measure the OS boundary itself.
    with tempfile.TemporaryDirectory(prefix="pf-host-sentinel-") as tmp:
        sentinel = Path(tmp) / "secret.txt"
        sentinel.write_text("host-sentinel-never-mounted")
        script = '''import json, os, socket
denied = False
try:
    socket.create_connection(("169.254.169.254",80), timeout=.2)
except OSError:
    denied = True
write_denied = False
try:
    open("/host-write", "w").write("bad")
except OSError:
    write_denied = True
value = {"uid":os.getuid(),"network_denied":denied,"rootfs_denied":write_denied,
"sentinel_absent":not os.path.exists(HOST_SENTINEL_PATH),"secret_env_absent":"PF_SECRET_SENTINEL" not in os.environ,
"gpu_absent":not os.path.exists("/dev/nvidia0"),"docker_absent":not os.path.exists("/var/run/docker.sock")}
print(json.dumps({"status":"OK","compiled":True,"returned":True,"value":value}))
'''.replace("HOST_SENTINEL_PATH", repr(str(sentinel)))
        previous = os.environ.get("PF_SECRET_SENTINEL")
        os.environ["PF_SECRET_SENTINEL"] = "temporary-test-value-never-forwarded"
        try:
            result = sandbox.invoke("", "repair(x)", {"args": [], "kwargs": {}}, _probe_script=script)
        finally:
            if previous is None:
                os.environ.pop("PF_SECRET_SENTINEL",None)
            else:
                os.environ["PF_SECRET_SENTINEL"] = previous
        value = result.get("value", {})
        for name in ("network_denied", "rootfs_denied", "sentinel_absent", "secret_env_absent", "gpu_absent", "docker_absent"):
            checks.append({"id": name, "passed": value.get(name) is True})
        checks.append({"id": "nonroot", "passed": value.get("uid") == 65532})
        checks.append({"id": "host-sentinel-unchanged", "passed": sentinel.read_text() == "host-sentinel-never-mounted"})
    def probe(script):
        return sandbox.invoke("","repair(x)",{"args":[],"kwargs":{}},_probe_script=script)
    output = probe("import sys\nsys.stdout.write('x' * (2 * 1024 * 1024))\nsys.stdout.flush()")
    checks.append({"id":"output-cap","passed":output["status"]=="RESOURCE_LIMIT"})
    first = probe("import json\nopen('/tmp/sentinel','w').write('private')\nprint(json.dumps({'status':'OK','compiled':True,'value':True}))")
    second = probe("import os,json\nprint(json.dumps({'status':'OK','compiled':True,'value':not os.path.exists('/tmp/sentinel')}))")
    checks.append({"id":"fresh-workspace","passed":first.get("value") is True and second.get("value") is True})
    memory = probe("import json\nx=bytearray(512*1024*1024)\nprint(json.dumps({'status':'OK','compiled':True,'value':False}))")
    checks.append({"id":"container-memory-cap","passed":memory["status"]=="RESOURCE_LIMIT"})
    processes = probe('''import os,time,json,signal
children=[]
limited=False
try:
    for i in range(64):
        try:
            pid=os.fork()
        except OSError:
            limited=True
            break
        if pid==0:
            time.sleep(10)
            os._exit(0)
        children.append(pid)
finally:
    for pid in children:
        try: os.kill(pid,signal.SIGKILL)
        except OSError: pass
    for pid in children:
        try: os.waitpid(pid,0)
        except OSError: pass
print(json.dumps({'status':'OK','compiled':True,'value':limited and len(children)<32}))
''')
    checks.append({"id":"process-cap","passed":processes.get("value") is True})
    remaining = subprocess.run(["docker","ps","-aq","--filter","label=python-fixer.owner=m1"],
        capture_output=True,text=True,env=sandbox.environment,timeout=10)
    checks.append({"id":"no-container-leaks","passed":remaining.returncode==0 and not remaining.stdout.strip()})
    return {"schema_version": 1, "scope": "real-runsc-cpu-preflight",
        "status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks,
        "image_digest": sandbox.policy.image, "policy_sha256": sandbox.policy.sha256,
        "limitations": ["not a kernel escape proof", "not an authenticated remote deployment", "no GPU or model test"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    try:
        result = run_preflight(DockerSandbox(Policy.load(Path(args.config))))
        if args.report:
            write_json(Path(args.report), result)
        print(f"{result['status']}: {sum(c['passed'] for c in result['checks'])}/{len(result['checks'])} actual CPU isolation checks")
        for case in result["checks"]:
            if not case["passed"]:
                print("FAIL:", case["id"])
        return 0 if result["status"] == "PASS" else 1
    except Exception as exc:
        if args.report:
            write_json(Path(args.report),{"schema_version":1,"scope":"real-runsc-cpu-preflight",
                "status":"BLOCKED","reason":str(exc),"checks":[],"candidate_execution":"NOT_RUN"})
        print("SANDBOX BLOCKED:", str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

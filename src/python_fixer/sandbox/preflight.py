"""Exercise actual gVisor isolation. No PASS is emitted without runtime checks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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
print(json.dumps({'status':'OK','compiled':True,'value':[limited, len(children)]}))
''')
    reached = processes.get("value")
    checks.append({"id":"process-cap","passed":isinstance(reached,list) and reached[0] is True and reached[1] < 32,
                   "status":processes["status"],"children_reached":reached[1] if isinstance(reached,list) else None})
    threads = probe('''import json,_thread,time
count=0
limited=False
try:
    for i in range(256):
        _thread.start_new_thread(time.sleep,(3,))
        count+=1
except Exception:
    limited=True
print(json.dumps({'status':'OK','compiled':True,'value':[limited,count]}))
''')
    # Bounded is required: either thread creation is refused inside the sandbox, or the
    # whole container is stopped by a limit. INTERNAL_ERROR (sandbox crash) is a failure.
    value = threads.get("value")
    checks.append({"id":"thread-storm-bounded","status":threads["status"],
                   "threads_reached":value[1] if isinstance(value,list) else None,
                   "passed":threads["status"]=="RESOURCE_LIMIT" or (threads["status"]=="OK" and isinstance(value,list) and value[0] is True)})
    dns = probe('''import json,socket
try:
    socket.getaddrinfo("example.com",80)
    denied=False
except OSError:
    denied=True
print(json.dumps({'status':'OK','compiled':True,'value':denied}))
''')
    checks.append({"id":"dns_denied","passed":dns.get("value") is True})
    began = time.monotonic()
    orphan = probe('''import os,json,time
if os.fork()==0:
    os.setsid()
    time.sleep(60)
    os._exit(0)
print(json.dumps({'status':'OK','compiled':True,'value':True}),flush=True)
''')
    elapsed = time.monotonic() - began
    leftover = subprocess.run(["docker","ps","-aq","--filter","label=python-fixer.owner=m1"],
        capture_output=True,text=True,env=sandbox.environment,timeout=10)
    checks.append({"id":"descendant-bounded","status":orphan["status"],"elapsed_s":round(elapsed,2),
                   "passed":orphan["status"] in {"OK","TIMEOUT"} and elapsed < sandbox.policy.wall_seconds + 15
                            and leftover.returncode==0 and not leftover.stdout.strip()})
    # Candidate-side failures must be ordinary candidate failures, not accepted reports.
    for name, code in (("exception","def repair(x):\n    raise ValueError('boom')\n"),
                       ("recursion","def repair(x):\n    return repair(x)\n"),
                       ("nested-return","def repair(x):\n    v = []\n    for i in range(200):\n        v = [v]\n    return v\n")):
        result = sandbox.invoke(code, "repair(x)", {"args": [1], "kwargs": {}})
        checks.append({"id":name,"status":result["status"],
                       "passed":result["status"] not in {"INTERNAL_ERROR"} and result.get("returned") is not True})
    # End to end through the trusted gateway: expected values never reach the sandbox.
    # The probe replaces the shim and *returns the sentinel if it can see it*, so a
    # leak would make the test pass; both the per-test and suite paths are exercised.
    from .gateway import execute_suite
    sentinel = "expected-sentinel-" + os.urandom(8).hex()
    leak_probe = '''import json,os,sys
raw=sys.stdin.read()
seen=SENTINEL in raw or any(SENTINEL in v for v in os.environ.values())
for root in ("/tmp","/"):
    try: names=os.listdir(root)
    except OSError: names=[]
    seen = seen or any(SENTINEL in n for n in names)
one={'status':'OK','compiled':True,'returned':True,'value':SENTINEL if seen else 'absent','cpu_ms':1,'peak_memory_bytes':1}
request=json.loads(raw)
if "tests" in request:
    print(json.dumps({'status':'OK','compiled':True,'results':[one for t in request["tests"]],'cpu_ms':1,'peak_memory_bytes':1}))
else:
    print(json.dumps(one))
'''.replace("SENTINEL", repr(sentinel))
    class Spy(DockerSandbox):
        observed = []
        def invoke(self, source, signature, test, **kw):
            kw["_probe_script"] = leak_probe
            outcome = super().invoke(source, signature, test, **kw)
            self.observed.append(outcome)
            return outcome
    class PerTest:
        def __init__(self, spy):
            self.policy, self.invoke = spy.policy, spy.invoke
    row = {"tests": [{"test_id": "1", "args": [1], "kwargs": {}, "expected": sentinel, "comparator": "exact"},
                     {"test_id": "2", "args": [2], "kwargs": {}, "expected": sentinel, "comparator": "exact"}],
           "signature": "repair(x)"}
    candidate = "def repair(x):\n    return x\n"
    for path, target in (("suite", Spy(sandbox.policy)), ("per-test", PerTest(Spy(sandbox.policy)))):
        Spy.observed = []
        result = execute_suite(target, row, candidate)
        values = [r.get("value") for o in Spy.observed for r in o.get("results", [o])]
        checks.append({"id": f"expected-values-absent-{path}", "status": result["status"], "values": values,
                       "passed": result["status"] == "OK" and result["passed_ids"] == []
                                 and len(values) == 2 and all(v == "absent" for v in values)})
    # Positive control: the same probe must report the sentinel when it *is* visible.
    control = sandbox.invoke("", sentinel, {"args": [1], "kwargs": {}}, _probe_script=leak_probe)
    checks.append({"id": "expected-probe-positive-control", "value_matches": control.get("value") == sentinel,
                   "passed": control.get("value") == sentinel})
    # A real segfault from pure candidate code (deeply nested tuple hash) must be a
    # candidate failure on both paths.
    crash = "def repair(x):\n    t = ()\n    for i in range(1000000):\n        t = (t,)\n    return len({t})\n"
    single = sandbox.invoke(crash, "repair(x)", {"args": [1], "kwargs": {}})
    suite = sandbox.invoke_suite(crash, "repair(x)", [{"args": [1], "kwargs": {}}])
    checks.append({"id": "segfault-candidate-failure", "single_status": single["status"],
                   "suite_status": [o.get("status") for o in suite],
                   "passed": single["status"] not in {"INTERNAL_ERROR"} and single.get("returned") is not True
                             and all(o.get("status") != "INTERNAL_ERROR" and o.get("returned") is not True for o in suite)})
    # Candidate return values that cannot be serialized must be candidate failures
    # on the production suite path, never INTERNAL_ERROR.
    for name, code in (("big-int-return", "def repair(x):\n    return pow(10, 5000)\n"),
                       ("surrogate-return", "def repair(x):\n    return '\\ud800'\n")):
        outcome = sandbox.invoke_suite(code, "repair(x)", [{"args": [1], "kwargs": {}}])
        suite = execute_suite(sandbox, {"tests": [{"test_id": "1", "args": [1], "kwargs": {}, "expected": 0,
                                                    "comparator": "exact"}], "signature": "repair(x)"}, code)
        checks.append({"id": name, "status": [o.get("status") for o in outcome], "suite_status": suite["status"],
                       "passed": suite["status"] != "INTERNAL_ERROR" and suite["passed_ids"] == []
                                 and all(o.get("status") != "INTERNAL_ERROR" for o in outcome)})
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

"""Bounded gVisor invocations with no mounts, inherited credentials or fallback."""
from __future__ import annotations
from pathlib import Path
import os
import subprocess
import tempfile
import time
import uuid
from python_fixer.common import canonical_bytes, strict_loads
from .policy import Policy, SandboxUnavailable


class DockerSandbox:
    def __init__(self, policy: Policy):
        self.policy = policy
        # Only transport/runtime configuration, never parent secret variables.
        self.environment = {k: v for k, v in os.environ.items()
                            if k in {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "SSL_CERT_FILE"}}

    def available(self):
        try:
            result = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                env=self.environment, capture_output=True, text=True, timeout=10)
            runtimes = strict_loads(result.stdout) if not result.returncode else {}
            if "runsc" not in runtimes:
                raise SandboxUnavailable("Docker daemon must expose runsc; no fallback allowed")
            flags = runtimes["runsc"].get("runtimeArgs",[])
            if any(flag in {"--ignore-cgroups","--ignore-cgroups=true"} for flag in flags):
                raise SandboxUnavailable("runsc ignores cgroups; real memory-limit probe failed. Configure an enforcing dedicated CPU worker before executing candidates")
            image = subprocess.run(["docker", "image", "inspect", self.policy.image],
                env=self.environment, capture_output=True, timeout=10)
            if image.returncode:
                raise SandboxUnavailable("pinned image is not locally installed")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise SandboxUnavailable(str(exc)) from exc

    def command(self, name: str, cpu_seconds: int) -> list[str]:
        p = self.policy
        return ["docker", "create", "--pull=never", "--name", name, "--label=python-fixer.owner=m1",
                "--runtime=runsc", "--network=none",
                "--read-only", "--user=65532:65532", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                f"--memory={p.memory_mib}m", f"--memory-swap={p.memory_mib}m", "--cpus=1",
                f"--pids-limit={p.pids}", f"--ulimit=cpu={cpu_seconds}:{cpu_seconds}",
                f"--ulimit=nproc={p.pids}:{p.pids}", "--ulimit=nofile=64:64", "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--log-driver=none", "-i", p.image, "python", "-I", "-B", "-c",
                (Path(__file__).with_name("child.py")).read_text()]

    def invoke(self, source: str, signature: str, test: dict, *, wall_seconds: float | None = None,
               cpu_seconds: int | None = None, _probe_script: str | None = None,
               _tests: list[dict] | None = None) -> dict:
        p = self.policy
        self.available()
        limit = min(p.wall_seconds, wall_seconds if wall_seconds is not None else p.wall_seconds)
        cpu = min(p.cpu_seconds, cpu_seconds if cpu_seconds is not None else p.cpu_seconds)
        if limit <= 0 or cpu <= 0:
            return {"status": "TIMEOUT", "compiled": False, "cpu_ms": 0, "peak_memory_bytes": 0}
        request = {"source": source, "signature": signature, "args": test["args"],
            "kwargs": test["kwargs"], "imports": p.imports, "cpu_seconds": cpu, "memory_mib": p.memory_mib}
        if _tests is not None:
            request.update(tests=_tests, wall_seconds=limit,
                           runner_source=Path(__file__).with_name("child.py").read_text())
        payload = canonical_bytes(request)
        if len(payload) > 131072:
            return {"status": "INVALID_OUTPUT", "compiled": False, "cpu_ms": 0, "peak_memory_bytes": 0}
        name = "pf-" + uuid.uuid4().hex
        process = None
        began = time.monotonic()
        outcome = {"status": "INTERNAL_ERROR", "compiled": False, "cpu_ms": 0, "peak_memory_bytes": 0}
        try:
            # Files bound host RAM usage even for hostile child output. Reads are capped.
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error, tempfile.TemporaryFile() as stdin:
                stdin.write(payload)
                stdin.seek(0)
                command = self.command(name, cpu)
                if _probe_script is not None:
                    command[-1] = _probe_script  # Operator-owned containment probes only; never exposed by gateway.
                created = subprocess.run(command,capture_output=True,env=self.environment,timeout=15)
                if created.returncode:
                    raise OSError("container creation failed")
                began = time.monotonic()
                process = subprocess.Popen(["docker","start","--attach","--interactive",name], stdin=stdin, stdout=output,
                    stderr=error, env=self.environment)
                while process.poll() is None:
                    if output.tell() + error.tell() > p.output_bytes:
                        outcome["status"] = "RESOURCE_LIMIT"
                        break
                    if time.monotonic() - began > limit:
                        outcome["status"] = "TIMEOUT"
                        break
                    time.sleep(0.01)
                else:
                    if output.tell()+error.tell()>p.output_bytes:
                        outcome["status"] = "RESOURCE_LIMIT"
                    elif process.returncode == 0:
                        output.seek(0)
                        raw = output.read(p.output_bytes+1)
                        if len(raw) <= p.output_bytes:
                            value = strict_loads(raw.decode())
                            if isinstance(value, dict) and value.get("status") in {"OK", "COMPILE_ERROR", "INVALID_OUTPUT", "POLICY_VIOLATION", "RESOURCE_LIMIT"} and type(value.get("compiled")) is bool:
                                outcome = value
                        else:
                            outcome["status"] = "RESOURCE_LIMIT"
                    elif process.returncode in {137, 152}:
                        outcome["status"] = "RESOURCE_LIMIT"
        except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired):
            pass
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            try:
                cleanup = subprocess.run(["docker", "rm", "--force", name], env=self.environment,
                    capture_output=True, timeout=10)
                if cleanup.returncode and b"No such container" not in cleanup.stderr:
                    outcome["status"] = "INTERNAL_ERROR"
            except (OSError, subprocess.TimeoutExpired):
                outcome["status"] = "INTERNAL_ERROR"
        outcome["wall_ms"] = int((time.monotonic()-began)*1000)
        return outcome

    def invoke_suite(self, source: str, signature: str, tests: list[dict]) -> list[dict]:
        result = self.invoke(source,signature,{"args":[],"kwargs":{}},_tests=tests)
        return result.get("results",[result])

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from python_fixer.common import digest, strict_loads


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    image: str
    wall_seconds: float = 5
    cpu_seconds: int = 2
    memory_mib: int = 256
    pids: int = 32
    # Host-side cgroup bound on the whole runsc sandbox (Sentry/Gofer threads plus
    # guest tasks). Under gVisor the candidate's 32-process limit is enforced inside
    # the sandbox via RLIMIT_NPROC; a host pids cgroup equal to that limit kills the
    # Sentry itself on a fork storm instead of returning EAGAIN (observed 2026-11-23).
    runtime_pids: int = 128
    output_bytes: int = 65536
    imports: tuple[str, ...] = ("math", "collections", "itertools", "functools", "heapq", "bisect", "re", "string")

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}", self.image):
            raise ValueError("sandbox image must use an immutable sha256 digest")
        if not (0 < self.wall_seconds <= 60 and 0 < self.cpu_seconds <= 30 and
                32 <= self.memory_mib <= 256 and 1 <= self.pids <= 32 and
                self.runtime_pids == 128 and
                1024 <= self.output_bytes <= 65536):
            raise ValueError("resource limits exceed approved policy")
        allowed = {"math", "collections", "itertools", "functools", "heapq", "bisect", "re", "string"}
        if not set(self.imports) <= allowed:
            raise ValueError("unapproved import capability")

    @property
    def sha256(self):
        from python_fixer.common import text_digest
        code = {name:text_digest(Path(__file__).with_name(name).read_text())
                for name in ("child.py","docker.py","gateway.py","policy.py","preflight.py")}
        return digest({"version": "python-pure-v1", **self.__dict__, "runner_code":code})

    @classmethod
    def load(cls, path: Path):
        data = strict_loads(path.read_text())
        if data["runtime"] != "runsc" or data["network"] != "none" or not data["read_only_rootfs"]:
            raise ValueError("runtime/network/filesystem policy cannot be weakened")
        if data["uid"] != 65532 or data["gid"] != 65532 or data["cap_drop"] != ["ALL"] or not data["no_new_privileges"]:
            raise ValueError("non-root and capability policy cannot be weakened")
        if data["cpus"] != 1 or data["swap_mib"] != 0 or data["tmpfs_mib"] != 16:
            raise ValueError("unexpected CPU/swap/tmpfs policy")
        return cls(image=data["image_digest"], wall_seconds=data["wall_timeout_seconds"],
                   cpu_seconds=data["cpu_timeout_seconds"], memory_mib=data["memory_mib"],
                   pids=data["pids_limit"], runtime_pids=data["runtime_host_pids_limit"],
                   output_bytes=data["output_limit_bytes"],
                   imports=tuple(data["allowed_candidate_imports"]))

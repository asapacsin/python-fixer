"""mTLS-only reward client. The trainer never receives Docker authority."""
from __future__ import annotations
import math
from pathlib import Path
import ssl
import re
import copy
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from jsonschema import Draft202012Validator
from python_fixer.common import canonical_bytes, strict_loads, text_digest, resource_path
from python_fixer.rewards import behavior_reward, InvalidOutcome
from .policy import SandboxUnavailable


class RewardClient:
    def __init__(self, endpoint: str, *, ca: str, certificate: str, key: str,
                 bindings: dict[str, dict], run_id: str, request_timeout: float = 30):
        if (not isinstance(request_timeout, (int, float))
                or not math.isfinite(request_timeout) or request_timeout <= 0):
            raise ValueError("request_timeout must be a finite positive number")
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("an HTTPS mTLS endpoint without embedded credentials is required")
        required = {"suite_sha256","image_digest","policy_sha256"}
        if not bindings:
            raise ValueError("frozen train bindings cannot be empty")
        for problem_id,entry in bindings.items():
            if not isinstance(problem_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}",problem_id):
                raise ValueError("invalid problem binding ID")
            if set(entry) != required or any(not isinstance(entry[k],str) for k in required):
                raise ValueError("complete frozen suite/image/policy bindings required")
            if not all(re.fullmatch(r"[0-9a-f]{64}",entry[k]) for k in ("suite_sha256","policy_sha256")):
                raise ValueError("invalid frozen digest")
            if not re.fullmatch(r"[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}",entry["image_digest"]):
                raise ValueError("image binding must be immutable")
        self.endpoint = endpoint.rstrip("/") + "/score"
        self.context = ssl.create_default_context(cafile=ca)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.context.load_cert_chain(certificate, key)
        # Avoid forwarding private worker traffic or client certificates to ambient proxies.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                         urllib.request.HTTPSHandler(context=self.context))
        self.bindings = copy.deepcopy(bindings)
        self.run_id = run_id
        self.request_timeout = float(request_timeout)
        self.validator = Draft202012Validator(strict_loads(resource_path("schemas/sandbox-outcome.schema.json").read_text()))

    def score(self, problem_id: str, source: str) -> tuple[float, dict]:
        if problem_id not in self.bindings:
            raise ValueError("problem not in frozen train bindings")
        request_id = str(uuid.uuid4())
        request = {"schema_version":1,"request_id":request_id,"run_id":self.run_id,
                   "problem_id":problem_id,"candidate_source":source,"candidate_sha256":text_digest(source),
                   "suite_role":"train_reward","limits_profile":"python-pure-v1"}
        body = canonical_bytes(request)
        if len(body)>131072 or len(source.encode())>65536:
            raise ValueError("candidate request exceeds limit")
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type":"application/json"})
                with self.opener.open(req, timeout=self.request_timeout) as response:
                    data = response.read(65537)
                if len(data)>65536:
                    raise InvalidOutcome("oversized gateway response")
                result = strict_loads(data.decode())
                self.validator.validate(result)
                expected = {**self.bindings[problem_id], "request_id":request_id,
                            "candidate_sha256":request["candidate_sha256"]}
                return behavior_reward(result, expected_bindings=expected), result
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    raise SandboxUnavailable("gateway rejected authorized request") from exc
            except (urllib.error.URLError, TimeoutError):
                pass
            if attempt < 2:
                time.sleep(.1 * 2**attempt)
        raise SandboxUnavailable("gateway unavailable after bounded retries")

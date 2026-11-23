"""Separate CPU worker entrypoint: TLS client identity owns train registry access."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import ssl
import sys
import threading
from python_fixer.common import canonical_bytes, strict_loads
from python_fixer.data.artifacts import load_verified_partition
from .docker import DockerSandbox
from .gateway import Gateway
from .policy import Policy, SandboxUnavailable


def server_for(address: tuple[str,int], gateway: Gateway, allowed_names: set[str]):
    if not allowed_names:
        raise ValueError("explicit client certificate identity allowlist required")
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass  # No request/candidate payloads or certificate details in logs.

        def respond(self, code, payload):
            data = canonical_bytes(payload)
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Connection","close")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                cert = self.connection.getpeercert()
                identities = {value for group in cert.get("subject",()) for key,value in group if key=="commonName"}
                if len(identities)!=1 or not identities <= allowed_names:
                    return self.respond(403,{"error":"client identity denied"})
                if self.path != "/score" or self.headers.get("Content-Type") != "application/json":
                    return self.respond(400,{"error":"invalid endpoint/content type"})
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths)!=1 or self.headers.get("Transfer-Encoding"):
                    return self.respond(400,{"error":"one bounded content length required"})
                length = int(lengths[0])
                if not 0 < length <= 131072:
                    return self.respond(413,{"error":"request size"})
                payload = self.rfile.read(length)
                if len(payload)!=length:
                    return self.respond(400,{"error":"incomplete body"})
                return self.respond(200,gateway.handle(strict_loads(payload.decode())))
            except (ValueError, TypeError, KeyError, UnicodeError, socket.timeout):
                return self.respond(400,{"error":"invalid request"})
            except SandboxUnavailable:
                return self.respond(503,{"error":"worker unavailable"})
            except Exception:
                return self.respond(500,{"error":"worker failure"})

    class BoundedServer(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 8
        def __init__(self,*args,**kwargs):
            self.slots = threading.BoundedSemaphore(8)
            super().__init__(*args,**kwargs)

        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request,address)
            except Exception:
                self.slots.release()
                raise

        def process_request_thread(self,request,address):
            try:
                super().process_request_thread(request,address)
            finally:
                self.slots.release()
        # Bound TLS handshakes / slow senders before allocating more threads.
        def get_request(self):
            raw, address = self.socket.accept()
            raw.settimeout(10)
            try:
                return self.tls_context.wrap_socket(raw, server_side=True), address
            except Exception:
                raw.close()
                raise

    return BoundedServer(address, Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, help="immutable dataset directory with manifest and evidence")
    parser.add_argument("--manifest-sha256",required=True,help="operator-pinned canonical manifest digest")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--client-name", action="append", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args(argv)
    try:
        sandbox = DockerSandbox(Policy.load(Path(args.config)))
        sandbox.available()
        rows,_ = load_verified_partition(Path(args.dataset),"train",expected_manifest_sha256=args.manifest_sha256)
        gateway = Gateway(sandbox, rows)
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=args.ca)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(args.certificate, args.key)
        with server_for((args.host,args.port), gateway, set(args.client_name)) as server:
            server.tls_context = context
            server.serve_forever()
    except Exception as exc:
        print("WORKER BLOCKED:",str(exc),file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

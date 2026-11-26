"""Real TLS transport tests with a protocol double, not candidate execution."""
from pathlib import Path
import ssl
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from python_fixer.sandbox.server import server_for
from python_fixer.sandbox.client import RewardClient


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        def run(*args):
            subprocess.run(["openssl",*args],cwd=cls.root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        run("req","-x509","-newkey","rsa:2048","-nodes","-days","1","-keyout","ca.key","-out","ca.crt","-subj","/CN=DisposableTestCA","-addext","basicConstraints=critical,CA:TRUE")
        for name in ("server","trainer","outsider"):
            run("req","-newkey","rsa:2048","-nodes","-keyout",name+".key","-out",name+".csr","-subj","/CN="+name)
            ext=cls.root/(name+".ext")
            ext.write_text("subjectAltName=IP:127.0.0.1\nextendedKeyUsage="+("serverAuth" if name=="server" else "clientAuth")+"\n")
            run("x509","-req","-in",name+".csr","-CA","ca.crt","-CAkey","ca.key","-CAcreateserial","-days","1","-out",name+".crt","-extfile",name+".ext")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        class StubGateway:
            def handle(self, request):
                return {"schema_version":1,"request_id":request["request_id"],
                    "candidate_sha256":request["candidate_sha256"],"suite_sha256":"a"*64,
                    "image_digest":"python@sha256:"+"b"*64,"policy_sha256":"c"*64,
                    "status":"OK","compiled":True,"fixed_failed":1,"total_failed":1,
                    "preserved_passed":1,"total_passed":1,"all_passed":True,"wall_ms":1,
                    "cpu_ms":1,"peak_memory_bytes":1,"truncated_output":False,
                    "runner_version":"protocol-double","diagnostics":""}
        self.server=server_for(("127.0.0.1",0),StubGateway(),{"trainer"})
        context=ssl.create_default_context(ssl.Purpose.CLIENT_AUTH,cafile=str(self.root/"ca.crt"))
        context.verify_mode=ssl.CERT_REQUIRED
        context.load_cert_chain(str(self.root/"server.crt"),str(self.root/"server.key"))
        self.server.tls_context=context
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint=f"https://127.0.0.1:{self.server.server_address[1]}"

    def client(self,name="trainer"):
        return RewardClient(self.endpoint,ca=str(self.root/"ca.crt"),
            certificate=str(self.root/(name+".crt")),key=str(self.root/(name+".key")),
            bindings={"p":{"suite_sha256":"a"*64,"image_digest":"python@sha256:"+"b"*64,"policy_sha256":"c"*64}},run_id="transport-test")

    def test_real_mutual_tls_client_and_binding(self):
        reward,report=self.client().score("p","def f(x):\n    return x\n")
        self.assertEqual(reward,1)
        self.assertEqual(report["runner_version"],"protocol-double")

    def test_valid_ca_wrong_identity_is_denied(self):
        from python_fixer.sandbox.policy import SandboxUnavailable
        with self.assertRaises(SandboxUnavailable):
            self.client("outsider").score("p","def f(x):\n    return x\n")

    def test_no_client_certificate_is_denied(self):
        context=ssl.create_default_context(cafile=str(self.root/"ca.crt"))
        with self.assertRaises((urllib.error.URLError, ssl.SSLError)):
            urllib.request.urlopen(urllib.request.Request(self.endpoint+"/score",data=b"{}"),context=context,timeout=2)

    def test_plain_http_client_configuration_rejected(self):
        with self.assertRaises(ValueError):
            RewardClient("http://127.0.0.1",ca="unused",certificate="unused",key="unused",bindings={},run_id="x")

    def test_caller_mutation_does_not_change_frozen_bindings(self):
        bindings={"p":{"suite_sha256":"a"*64,"image_digest":"python@sha256:"+"b"*64,"policy_sha256":"c"*64}}
        client=RewardClient(self.endpoint,ca=str(self.root/"ca.crt"),certificate=str(self.root/"trainer.crt"),
            key=str(self.root/"trainer.key"),bindings=bindings,run_id="test")
        bindings["p"].clear()
        self.assertEqual(set(client.bindings["p"]),{"suite_sha256","image_digest","policy_sha256"})
        self.assertEqual(client.score("p","def f(x):\n    return x\n")[0],1)

    def test_missing_frozen_binding_rejected_before_tls(self):
        with self.assertRaisesRegex(ValueError,"complete frozen"):
            RewardClient(self.endpoint,ca="unused",certificate="unused",key="unused",bindings={"p":{}},run_id="test")

    def test_extended_worker_request_timeout_is_configurable(self):
        client = RewardClient(self.endpoint, ca=str(self.root/"ca.crt"),
            certificate=str(self.root/"trainer.crt"), key=str(self.root/"trainer.key"),
            bindings={"p":{"suite_sha256":"a"*64,"image_digest":"python@sha256:"+"b"*64,"policy_sha256":"c"*64}},
            run_id="extended-timeout", request_timeout=90)
        self.assertEqual(client.request_timeout, 90)
        self.assertEqual(client.score("p", "def f(x):\n    return x\n")[0], 1)

    def test_invalid_request_timeout_rejected(self):
        for timeout in (0, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "finite positive number"):
                RewardClient(self.endpoint, ca="unused", certificate="unused", key="unused",
                    bindings={}, run_id="test", request_timeout=timeout)


if __name__=="__main__":
    unittest.main()

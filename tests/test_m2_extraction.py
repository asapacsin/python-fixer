import hashlib
import unittest

from python_fixer.evaluation.extraction import MAX_MODULE_BYTES, TURN_END, extract_module

MODULE = "def add(a, b):\n    return a + b\n"


class ExtractionTests(unittest.TestCase):
    def test_valid_module_is_exact_bytes(self):
        result = extract_module(MODULE)
        self.assertEqual(result.status, "VALID")
        self.assertEqual(result.module, MODULE.encode())
        self.assertEqual(result.sha256, hashlib.sha256(MODULE.encode()).hexdigest())

    def test_single_trailing_turn_token_removed(self):
        self.assertEqual(extract_module(MODULE + TURN_END).module, MODULE.encode())
        self.assertEqual(extract_module(MODULE + TURN_END + TURN_END).reason, "extra_turn_token")
        self.assertEqual(extract_module(TURN_END + MODULE).reason, "extra_turn_token")

    def test_fences_and_thinking_rejected(self):
        for text in ("```python\n" + MODULE + "```", "~~~\n" + MODULE, "<think>x</think>\n" + MODULE,
                     MODULE + "<|im_start|>assistant", MODULE + "<|endoftext|>",
                     "<tool_call>{}</tool_call>\n" + MODULE, "<|fim_prefix|>" + MODULE):
            with self.subTest(text=text):
                result = extract_module(text)
                self.assertEqual((result.status, result.reason), ("INVALID", "forbidden_marker"))
                self.assertIsNone(result.module)

    def test_encoding_and_empty_rejected(self):
        self.assertEqual(extract_module(b"\xff\xfe").reason, "not_utf8")
        self.assertEqual(extract_module("x\ud800").reason, "not_utf8")
        self.assertEqual(extract_module(MODULE + "\x00").reason, "nul_byte")
        self.assertEqual(extract_module("a�").reason, "replacement_character")
        self.assertEqual(extract_module(" \n\t" + TURN_END).reason, "empty")

    def test_size_limit_is_inclusive(self):
        self.assertEqual(extract_module("#" * MAX_MODULE_BYTES).status, "VALID")
        self.assertEqual(extract_module("#" * (MAX_MODULE_BYTES + 1)).reason, "too_large")
        self.assertEqual(extract_module("é" * (MAX_MODULE_BYTES // 2 + 1)).reason, "too_large")

    def test_never_executes_or_compiles(self):
        # Syntactically invalid or hostile text is only bytes here; the worker judges it.
        result = extract_module("import os\nos.system('touch /tmp/pwn')\n")
        self.assertEqual(result.status, "VALID")
        self.assertEqual(extract_module("def (:\n").status, "VALID")

    def test_type_checked(self):
        with self.assertRaises(TypeError):
            extract_module(None)


if __name__ == "__main__":
    unittest.main()

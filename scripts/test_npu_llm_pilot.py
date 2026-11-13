import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import unittest

MODULE = Path(__file__).with_name("npu_llm_pilot.py")


class NpuPilotPolicy(unittest.TestCase):
    def test_an_unverified_compute_recipe_is_refused(self):
        self.assertTrue(MODULE.is_file(), "NPU-only pilot client is missing")
        spec = importlib.util.spec_from_file_location("npu_llm_pilot", MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.require_npu_recipe({"recipe": "llamacpp"})
        with self.assertRaises(ValueError):
            module.require_npu_recipe({})
        module.require_npu_recipe({"recipe": "flm"})

    def test_a_remote_endpoint_is_refused(self):
        spec = importlib.util.spec_from_file_location("npu_llm_pilot", MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "require_loopback"), "Loopback-only endpoint policy is missing")
        with self.assertRaises(ValueError):
            module.require_loopback("https://example.com/api/v1")
        with self.assertRaises(ValueError):
            module.require_loopback("http://user:secret@127.0.0.1:13305/api/v1")
        module.require_loopback("http://127.0.0.1:13305/api/v1")


@unittest.skipUnless(os.getenv("GRAMGRID_LIVE_NPU_PILOT") == "1", "requires the explicitly loaded NPU pilot")
class LiveNpuPilot(unittest.TestCase):
    def test_the_real_model_produces_a_npu_receipt(self):
        import json
        result = subprocess.run([sys.executable, str(MODULE), "--max-tokens", "64"], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip(), "Model client did not emit a real receipt")
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["recipe"], "flm")
        self.assertEqual(receipt["checkpoint"], "qwen3.5:0.8b")
        self.assertTrue(receipt["text"].strip())
        self.assertFalse(receipt["distributed_model_inference"])
        self.assertGreater(receipt["usage"]["completion_tokens"], 0)
        self.assertIsInstance(receipt["ttft_seconds"], (int, float))
        self.assertIsInstance(receipt["decode_tokens_per_second"], (int, float))


if __name__ == "__main__":
    unittest.main()

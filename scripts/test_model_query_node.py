import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/gramgrid-node/examples/model_query.rs"
BINARY = ROOT / "target/debug/examples/model_query"


class RealModelQueryNode(unittest.TestCase):
    def test_a_real_npu_model_answers_a_gg02_query_once(self):
        self.assertTrue(SOURCE.is_file(), "GG02 NPU model-query node is missing")
        self.assertTrue(BINARY.is_file(), "GG02 NPU model-query node is not built")
        result = subprocess.run([str(BINARY), "--consent"], cwd=ROOT, capture_output=True, text=True, timeout=150)
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads(result.stdout)
        self.assertEqual(proof["model_calls"], 1)
        self.assertEqual(proof["replay_model_calls"], 0)
        self.assertTrue(proof["text"].strip())
        self.assertTrue(proof["replay_same_response"])
        self.assertTrue(proof["changed_input_refused"])
        self.assertTrue(proof["foreign_session_refused"])
        self.assertTrue(proof["foreign_revision_refused"])
        self.assertFalse(proof["model_layer_sharding"])
        self.assertEqual(proof["physical_npu_devices"], 1)
        self.assertFalse(proof["public_listener"])
        self.assertEqual(proof["recipe"], "flm")
        self.assertEqual(proof["checkpoint"], "qwen3.5:0.8b")
        (ROOT / "docs/evidence/gg02-model-query.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(proof, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()

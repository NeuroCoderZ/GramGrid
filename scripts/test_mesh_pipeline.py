import hashlib
import json
from pathlib import Path
import struct
import subprocess
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/debug/examples/mesh_pipeline"

class RealMeshPipeline(unittest.TestCase):
    def test_a_mismatched_fixture_manifest_is_rejected(self):
        self.assertTrue(BINARY.is_file())
        result = subprocess.run([str(BINARY), "f" * 64], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0, "accepted a manifest unrelated to resident weights")
        self.assertIn("manifest digest mismatch", result.stderr)

    def test_real_npu_peers_forward_activations_without_resending_weights(self):
        self.assertTrue(BINARY.is_file(), "GG02 NPU pipeline executable is not built")
        a = np.array([[2 if r == c else (1 if (r + c) % 7 == 0 else 0) for c in range(16)] for r in range(16)], dtype=np.int32)
        b = np.array([[3 if r == c else (-1 if (r * 3 + c) % 11 == 0 else 0) for c in range(16)] for r in range(16)], dtype=np.int32)
        digest = hashlib.sha256(a.astype("<i4").tobytes() + b.astype("<i4").tobytes()).hexdigest()
        result = subprocess.run([str(BINARY), digest], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["manifest_id"], digest)
        self.assertEqual(report["logical_npu_peers"], 2)
        self.assertEqual(report["physical_npu_devices"], 1)
        self.assertEqual(report["weight_loads"], [1, 1])
        self.assertEqual(report["state_handoff_bytes_preserved"], True)
        self.assertEqual(report["payload_bytes"], 1024)
        self.assertEqual(report["replay_without_recompute"], True)
        self.assertEqual(report["public_listener"], False)
        self.assertEqual(report["model_inference"], False)
        x = np.array(report["last_input"], dtype=np.int32).reshape(16, 16)
        expected = ((x.astype(np.int64) @ a.astype(np.int64)) @ b.astype(np.int64)).astype(np.int32)
        np.testing.assert_array_equal(np.array(report["last_output"], dtype=np.int32).reshape(16, 16), expected)
        self.assertEqual(report["all_runs_verified"], True)
        self.assertEqual(report["executions_per_peer"], [report["runs"] + 1] * 2)
        target = ROOT / "docs/evidence/npu-mesh-pipeline.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("last_input", "last_output")}, ensure_ascii=False))

if __name__ == "__main__":
    unittest.main()

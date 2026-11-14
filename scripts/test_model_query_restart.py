import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/debug/examples/model_query"


class ModelNodeRestart(unittest.TestCase):
    def test_a_completed_reply_survives_a_new_node_process_without_generation(self):
        state = tempfile.mkdtemp(prefix="gramgrid-journal-fixture-", dir=os.environ["TMPDIR"])
        command = [str(BINARY), "--consent", "--state-dir", state]
        first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=150)
        self.assertEqual(first.returncode, 0, "Persistent model-node invocation is missing: " + first.stderr)
        one = json.loads(first.stdout)
        second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=150)
        self.assertEqual(second.returncode, 0, second.stderr)
        two = json.loads(second.stdout)
        self.assertEqual(one["model_calls"], 1)
        self.assertEqual(two["model_calls"], 0)
        self.assertTrue(two["persistent_replay_after_restart"])
        self.assertEqual(one["text"], two["text"])
        self.assertEqual(one["response_digest"], two["response_digest"])
        self.assertFalse(two["model_layer_sharding"])
        proof = {"first": one, "after_restart": two, "fixture_owner": "GramGrid test; managed scratch TTL"}
        (ROOT / "docs/evidence/model-query-restart.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"calls_before": one["model_calls"], "calls_after": two["model_calls"], "same_response": one["response_digest"] == two["response_digest"]}))


if __name__ == "__main__":
    unittest.main()

"""Отдельные процессы клиента и сервера; NPU вызывается только в явном живом тесте."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import unittest
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "target/debug/examples/model_query"
CLIENT = ROOT / "target/debug/examples/model_query_client"


class ClientResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str
    pid: int


def start_server(state):
    process = subprocess.Popen(
        [str(SERVER), "--consent", "--state-dir", str(state), "--serve"],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout is not None and process.stderr is not None and process.stdin is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(8):
                raise AssertionError("Сервер не сообщил готовность отдельного исполнителя")
        line = process.stdout.readline()
        if not line:
            raise AssertionError("Режим отдельного сервера отсутствует: " + process.stderr.read())
        ready = json.loads(line)
        assert ready["journal_enabled"] is True
        assert ready["public_listener"] is False
        assert ready["listen"].startswith("127.0.0.1:")
        return process, ready
    except BaseException:
        process.kill()
        process.communicate(timeout=5)
        raise


def stop_server(process):
    stdout, stderr = process.communicate("stop\n", timeout=10)
    if process.returncode != 0:
        raise AssertionError(stderr)
    return json.loads(stdout.strip())


def client_request(ready, session=None, task=1, prompt="Ответь одним словом: РАБОТАЕТ"):
    command = [str(CLIENT), "--address", ready["listen"],
               "--session", str(ready["session_tag"] if session is None else session),
               "--revision", ready["model_revision"], "--task", str(task),
               "--max-tokens", "32", "--prompt", prompt]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=120)
    except BaseException:
        process.kill()
        process.communicate(timeout=5)
        raise
    assert process.returncode is not None
    return ClientResult(process.returncode, stdout, stderr, process.pid)


class ModelServiceProcess(unittest.TestCase):
    def test_serve_requires_persistent_state(self):
        result = subprocess.run([str(SERVER), "--consent", "--serve"], cwd=ROOT,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=8)
        self.assertNotEqual(result.returncode, 0, "Сервис допустил потерю ответов в режиме без журнала")
        self.assertIn("state-dir", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_a_separate_client_rejection_does_not_stop_the_server(self):
        state = tempfile.mkdtemp(prefix="gramgrid-model-service-", dir=os.environ["TMPDIR"])
        process, ready = start_server(state)
        try:
            result = client_request(ready, session=ready["session_tag"] ^ 1 or 2)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("session", result.stderr)
            self.assertIsNone(process.poll(), "Отказ одного клиента остановил сервер")
            summary = stop_server(process)
            self.assertEqual(summary["model_calls"], 0)
            self.assertEqual(summary["refused_requests"], 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


@unittest.skipUnless(os.getenv("GRAMGRID_LIVE_NPU_PILOT") == "1", "Нужен явно разрешённый NPU-пилот")
class LiveModelService(unittest.TestCase):
    def test_live_reply_is_reused_after_restarting_the_server(self):
        state = tempfile.mkdtemp(prefix="gramgrid-live-service-", dir=os.environ["TMPDIR"])
        first, info = start_server(state)
        try:
            one = client_request(info)
            self.assertEqual(one.returncode, 0, one.stderr)
            payload = json.loads(one.stdout)
            self.assertTrue(payload["text"].strip())
            self.assertEqual(payload["recipe"], "flm")
            self.assertNotEqual(one.pid, first.pid)
            replay = client_request(info)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual(json.loads(replay.stdout), payload)
            self.assertNotEqual(replay.pid, one.pid)
            before = stop_server(first)
            self.assertEqual(before["model_calls"], 1)
        finally:
            if first.poll() is None:
                first.kill()
                first.communicate(timeout=5)
        restarted, again = start_server(state)
        try:
            self.assertEqual(info["session_tag"], again["session_tag"])
            self.assertNotEqual(first.pid, restarted.pid)
            two = client_request(again)
            self.assertEqual(two.returncode, 0, two.stderr)
            self.assertEqual(json.loads(two.stdout), payload)
            after = stop_server(restarted)
            self.assertEqual(after["model_calls"], 0)
        finally:
            if restarted.poll() is None:
                restarted.kill()
                restarted.communicate(timeout=5)
        evidence = {"first_server": before, "after_restart": after,
                    "client_pids": [one.pid, replay.pid, two.pid],
                    "text": payload["text"], "checkpoint": payload["checkpoint"],
                    "recipe": payload["recipe"], "usage": payload["usage"],
                    "same_saved_receipt": True, "physical_npu_devices": 1,
                    "public_listener": False, "model_layer_sharding": False,
                    "fixture_owner": "GramGrid; managed scratch retention"}
        (ROOT / "docs/evidence/model-service-processes.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()

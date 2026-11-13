"""NPU-only model pilot through Lemonade; never a model-sharding engine."""

import argparse
from collections.abc import Mapping
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit
import urllib.request


def require_loopback(endpoint: str) -> None:
    url = urlsplit(endpoint)
    try:
        local = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        local = False
    if url.scheme not in ("http", "https") or not local or url.username is not None or url.password is not None or url.query or url.fragment:
        raise ValueError("Only literal loopback endpoints without embedded credentials are allowed")


def require_npu_recipe(model: Mapping[str, Any]) -> None:
    if model.get("recipe") != "flm":
        raise ValueError("Refusing an unknown or non-NPU compute recipe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:13305/api/v1")
    parser.add_argument("--model", default="user.GramGrid-Qwen35-08B-NPU")
    parser.add_argument("--prompt", default="Ответь одним словом: РАБОТАЕТ")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require_loopback(args.endpoint)
    if not 1 <= args.max_tokens <= 512 or not args.prompt or len(args.prompt.encode("utf-8")) > 4096:
        raise ValueError("Pilot request exceeds its bounded input or output budget")
    url = urlsplit(args.endpoint)
    key = os.getenv("LEMONADE_API_KEY", "local")
    env = dict(os.environ, LEMONADE_API_KEY=key)
    result = subprocess.run([
        "lemonade", "--host", str(url.hostname), "--port", str(url.port or 13305),
        "export", args.model,
    ], env=env, capture_output=True, text=True, check=True, timeout=15)
    model = json.loads(result.stdout)
    require_npu_recipe(model)
    checkpoint = model.get("checkpoints", {}).get("main")
    if checkpoint != "qwen3.5:0.8b" or model.get("model_name") != args.model:
        raise ValueError("This measured pilot admits only the exact Qwen3.5-0.8B NPU2 checkpoint")
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
                         "max_tokens": args.max_tokens, "temperature": 0, "stream": False}).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    # A local API must not redirect this credential-bearing request elsewhere.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Redirects are not permitted in the NPU pilot")
    started = time.perf_counter()
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=90) as response:
        raw = response.read(1048577)
    elapsed = time.perf_counter() - started
    if len(raw) > 1048576:
        raise ValueError("Model response exceeds the bounded receipt size")
    data = json.loads(raw)
    text = data["choices"][0]["message"].get("content")
    usage = data.get("usage", {})
    if not isinstance(text, str) or not text.strip() or not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] <= 0:
        raise ValueError("Model did not produce a verifiable nonempty text completion")
    receipt = {
        "model": args.model, "checkpoint": checkpoint, "recipe": model["recipe"],
        "text": text, "usage": usage, "wall_seconds": elapsed,
        "end_to_end_completion_tokens_per_second": usage["completion_tokens"] / elapsed,
        "ttft_seconds": usage.get("prefill_duration_ttft"),
        "decode_tokens_per_second": usage.get("decoding_speed_tps"),
        "timings_from_response": data.get("timings"),
        "npu_recipe_verified": True, "hardware_execution_verified_by_this_client": False,
        "distributed_model_inference": False,
        "scope": "One full native NPU2 model through Lemonade; hardware and power evidence are separate.",
    }
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NPU pilot failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

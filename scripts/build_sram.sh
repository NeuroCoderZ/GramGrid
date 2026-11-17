#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
import os
from pathlib import Path
tmp_value = os.environ.get("TMPDIR")
if not tmp_value:
    raise SystemExit("Set TMPDIR to an existing temporary directory")
for path in (Path.cwd(), Path(tmp_value)):
    state = os.statvfs(path)
    if state.f_bavail * state.f_frsize < 10 * 1024**3 + 128 * 1024**2 or state.f_favail < 1000:
        raise SystemExit("Insufficient storage reserve")
library = Path.cwd() / "native/xdna/build/libgramgrid_xdna_sram.so"
for process in Path("/proc").iterdir():
    if not process.name.isdigit():
        continue
    try:
        text = (process / "maps").read_text()
    except OSError:
        continue
    if str(library) in text:
        raise SystemExit(f"SRAM library is mapped by PID {process.name}; do not overwrite it")
PY
[[ -x .venv/bin/python ]] || { printf '%s\n' 'Existing NPU environment is required' >&2; exit 1; }
.venv/bin/python scripts/npu_memory_profile.py
.venv/bin/python native/xdna/build_sram_matmul.py
g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared native/xdna/adapter.cpp -lxrt_coreutil -o native/xdna/build/libgramgrid_xdna_sram.so
.venv/bin/python scripts/test_npu_sram_batch.py

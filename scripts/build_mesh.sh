#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
import os
from pathlib import Path
root = Path.cwd()
tmp_value = os.environ.get("TMPDIR")
if not tmp_value:
    raise SystemExit("Set TMPDIR to an existing temporary directory")
tmp = Path(tmp_value)
for location in (root, tmp):
    state = os.statvfs(location)
    if state.f_bavail * state.f_frsize < 10 * 1024**3 + 128 * 1024**2 or state.f_favail < 1000:
        raise SystemExit("Insufficient storage reserve for the bounded mesh build")
library = root / "native/xdna/build/libgramgrid_xdna_mesh.so"
for process in Path("/proc").iterdir():
    if not process.name.isdigit():
        continue
    try:
        maps = (process / "maps").read_text()
    except (OSError, PermissionError):
        continue
    if str(library) in maps:
        raise SystemExit(f"Mesh library is mapped by PID {process.name}; do not overwrite a live library")
PY
[[ -x .venv/bin/python ]] || { printf '%s\n' 'Existing NPU environment is required' >&2; exit 1; }
[[ -f native/xdna/build/matmul16.xclbin && -f native/xdna/build/matmul16.insts.bin ]] || {
    printf '%s\n' 'Existing compiled MATMUL16_I32 artifacts are required' >&2; exit 1;
}
g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared native/xdna/adapter.cpp -lxrt_coreutil -o native/xdna/build/libgramgrid_xdna_mesh.so
cargo build --offline -p gramgrid-node --example mesh_pipeline -j 1
.venv/bin/python scripts/test_npu_resident.py
.venv/bin/python scripts/test_mesh_pipeline.py
cargo test --offline --workspace -j 1

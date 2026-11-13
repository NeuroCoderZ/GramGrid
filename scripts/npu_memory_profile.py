"""2026-11-13 07:50 MSK — Report the installed MLIR-AIE memory layout, not free hardware SRAM."""
from pathlib import Path
import json
from aie.dialects.aie import AIEDevice, get_target_model

ROOT = Path(__file__).resolve().parents[1]
rows = []
for device in (AIEDevice.npu2, AIEDevice.npu2_1col):
    target = get_target_model(device)
    cores = sum(target.is_core_tile(c, r) for c in range(target.columns()) for r in range(target.rows()))
    memories = sum(target.is_mem_tile(c, r) for c in range(target.columns()) for r in range(target.rows()))
    local_bytes = target.get_local_memory_size()
    memtile_bytes = target.get_mem_tile_size()
    total = cores * local_bytes + memories * memtile_bytes
    rows.append({
        "target": str(device), "columns": target.columns(), "rows": target.rows(),
        "compute_tiles": cores, "memory_tiles": memories,
        "data_bytes_per_compute_tile": local_bytes,
        "data_bytes_per_memory_tile": memtile_bytes,
        "aggregate_data_bytes": total, "aggregate_data_MiB": total / 2**20,
        "capacity_kind": "compiler architecture data-memory definition",
        "measured_free_hardware_bytes": None,
    })
report = {
    "source": "installed MLIR-AIE target-model API",
    "targets": rows,
    "important": [
        "Tile memories are distributed, not one flat freely allocatable heap.",
        "Capacity is not the usable/free budget of a running graph or another device.",
        "Data SRAM excludes separate program memory and registers.",
        "HOST_ONLY buffers in the current GramGrid adapter are host-memory buffers, not on-chip SRAM.",
        "Current pipeline fills matrix_b for every run; it does not prove persistent SRAM weights.",
    ],
    "current_i32_tensor_bytes": 16 * 16 * 4,
    "three_tensor_bytes_without_fifo_replication_or_other_overheads": 3 * 16 * 16 * 4,
}
assert rows[0]["compute_tiles"] == 32 and rows[0]["memory_tiles"] == 8
assert rows[0]["aggregate_data_bytes"] == 6 * 2**20
assert report["three_tensor_bytes_without_fifo_replication_or_other_overheads"] < local_bytes
out = ROOT / "docs/evidence/npu-memory-layout.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))

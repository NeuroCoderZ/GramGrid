"""Compile the SRAM-held four-matrix profile in the existing NPU environment."""
from pathlib import Path
import os
import json

root = Path(__file__).resolve().parents[2]
os.environ["NPU_CACHE_HOME"] = str(root / "cache/npu")
os.environ["PATH"] = str(root / "toolchains/xrt-utils/usr/bin") + os.pathsep + os.environ["PATH"]
from matmul_sram_design import matmul_sram4

build = root / "native/xdna/build"
build.mkdir(parents=True, exist_ok=True)
xclbin, instructions = matmul_sram4.specialize().compile(
    xclbin_path=build / "matmul_sram4.xclbin",
    inst_path=build / "matmul_sram4.insts.bin")
print(json.dumps({"profile": "MATMUL16_I32_SRAM_BATCH4", "xclbin": str(xclbin),
                  "instructions": str(instructions), "iterations": 4,
                  "held_weight_objects": 1, "uses_host_input_output_buffers": True}))

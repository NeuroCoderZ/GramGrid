from pathlib import Path
import os
import json

root = Path(__file__).resolve().parents[2]
os.environ['NPU_CACHE_HOME'] = str(root / 'cache' / 'npu')
os.environ['PATH'] = str(root / 'toolchains/xrt-utils/usr/bin') + os.pathsep + os.environ['PATH']
from matmul_design import matmul16_i32

build = root / 'native/xdna/build'
build.mkdir(parents=True, exist_ok=True)
xclbin, instructions = matmul16_i32.specialize().compile(
    xclbin_path=build / 'matmul16.xclbin', inst_path=build / 'matmul16.insts.bin')
print(json.dumps({'operator': 'MATMUL16_I32', 'xclbin': str(xclbin),
                  'instructions': str(instructions)}, ensure_ascii=False))

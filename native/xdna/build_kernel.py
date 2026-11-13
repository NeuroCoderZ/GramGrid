from pathlib import Path
import json
import os

root = Path(__file__).resolve().parents[2]
os.environ['NPU_CACHE_HOME'] = str(root / 'cache' / 'npu')
from add_design import add_i32

build = root / 'native' / 'xdna' / 'build'
build.mkdir(parents=True, exist_ok=True)
design = add_i32.specialize(count=1024)
xclbin, instructions = design.compile(
    xclbin_path=build / 'add_i32.xclbin',
    inst_path=build / 'add_i32.insts.bin',
)
print(json.dumps({'xclbin': str(xclbin), 'instructions': str(instructions),
                  'operator': 'ADD_I32', 'elements_per_dispatch': 1024}, ensure_ascii=False))

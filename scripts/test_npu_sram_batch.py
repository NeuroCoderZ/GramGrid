import ctypes
import json
from pathlib import Path
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "native/xdna/build"

class SramBatch(unittest.TestCase):
    def test_four_forwards_share_one_held_on_chip_weight_tile(self):
        library = BUILD / "libgramgrid_xdna_sram.so"
        self.assertTrue(library.is_file(), "SRAM batch kernel interface is not built")
        lib = ctypes.CDLL(str(library))
        ptr = ctypes.POINTER(ctypes.c_int32)
        lib.gramgrid_sram_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.gramgrid_sram_create.restype = ctypes.c_void_p
        lib.gramgrid_sram_load_weights.argtypes = [ctypes.c_void_p, ptr, ctypes.c_uint32]
        lib.gramgrid_sram_load_weights.restype = ctypes.c_int
        lib.gramgrid_sram_forward4.argtypes = [ctypes.c_void_p, ptr, ptr, ctypes.c_uint32]
        lib.gramgrid_sram_forward4.restype = ctypes.c_int
        lib.gramgrid_sram_error.restype = ctypes.c_char_p
        lib.gramgrid_sram_destroy.argtypes = [ctypes.c_void_p]
        lib.gramgrid_sram_device_name.argtypes = [ctypes.c_void_p]
        lib.gramgrid_sram_device_name.restype = ctypes.c_char_p
        handle = lib.gramgrid_sram_create(str(BUILD / "matmul_sram4.xclbin").encode(), str(BUILD / "matmul_sram4.insts.bin").encode())
        self.assertTrue(handle, lib.gramgrid_sram_error())
        try:
            rng = np.random.default_rng(130)
            x = rng.integers(-128, 128, (4, 16, 16), dtype=np.int32)
            weights = rng.integers(-8, 8, (16, 16), dtype=np.int32)
            out = np.zeros_like(x)
            def pointer(value):
                return value.ctypes.data_as(ptr)
            self.assertNotEqual(lib.gramgrid_sram_forward4(handle, pointer(x), pointer(out), 1024), 0)
            self.assertEqual(lib.gramgrid_sram_load_weights(handle, pointer(weights), 256), 0, lib.gramgrid_sram_error())
            self.assertNotEqual(lib.gramgrid_sram_forward4(handle, pointer(x), pointer(out), 1023), 0)
            durations = []
            for _ in range(3):
                start = time.perf_counter_ns()
                self.assertEqual(lib.gramgrid_sram_forward4(handle, pointer(x), pointer(out), 1024), 0, lib.gramgrid_sram_error())
                durations.append(time.perf_counter_ns() - start)
                np.testing.assert_array_equal(out, x @ weights)
            report = {
                "device": lib.gramgrid_sram_device_name(handle).decode(),
                "passed": True,
                "scope": "weights held on chip across four iterations of one NPU kernel invocation",
                "matrices_per_call": 4,
                "kernel_calls": 3,
                "elements_checked_per_call": int(out.size),
                "weight_tile_bytes": int(weights.nbytes),
                "host_input_bytes": int(x.nbytes),
                "host_output_bytes": int(out.nbytes),
                "host_DDR_still_used": True,
                "per_call_elapsed_ns": durations,
                "measured_hardware_DDR_traffic_bytes": None,
                "no_claim_of_SRAM_persistence_across_kernel_reset": True,
            }
            (ROOT / "docs/evidence/npu-sram-batch4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(report, ensure_ascii=False))
        finally:
            lib.gramgrid_sram_destroy(handle)

if __name__ == "__main__":
    unittest.main()

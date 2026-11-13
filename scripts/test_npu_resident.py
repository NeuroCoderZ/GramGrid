import ctypes
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "native/xdna/build"

class ResidentNpu(unittest.TestCase):
    def test_one_weight_load_serves_multiple_real_npu_forwards(self):
        library = BUILD / "libgramgrid_xdna_mesh.so"
        self.assertTrue(library.is_file(), "Resident NPU interface is not built")
        lib = ctypes.CDLL(str(library))
        ptr = ctypes.POINTER(ctypes.c_int32)
        lib.gramgrid_mesh_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.gramgrid_mesh_create.restype = ctypes.c_void_p
        lib.gramgrid_mesh_error.restype = ctypes.c_char_p
        lib.gramgrid_mesh_load_weights.argtypes = [ctypes.c_void_p, ptr, ctypes.c_uint32]
        lib.gramgrid_mesh_load_weights.restype = ctypes.c_int
        lib.gramgrid_mesh_forward.argtypes = [ctypes.c_void_p, ptr, ptr, ctypes.c_uint32]
        lib.gramgrid_mesh_forward.restype = ctypes.c_int
        lib.gramgrid_mesh_weight_loads.argtypes = [ctypes.c_void_p]
        lib.gramgrid_mesh_weight_loads.restype = ctypes.c_uint64
        lib.gramgrid_mesh_executions.argtypes = [ctypes.c_void_p]
        lib.gramgrid_mesh_executions.restype = ctypes.c_uint64
        lib.gramgrid_mesh_destroy.argtypes = [ctypes.c_void_p]
        handle = lib.gramgrid_mesh_create(str(BUILD / "matmul16.xclbin").encode(), str(BUILD / "matmul16.insts.bin").encode())
        self.assertTrue(handle, lib.gramgrid_mesh_error())
        try:
            x = (np.arange(256, dtype=np.int32) - 128).reshape(16, 16)
            weights = (np.eye(16, dtype=np.int32) * 2).copy()
            out = np.zeros((16, 16), dtype=np.int32)
            pointer = lambda value: value.ctypes.data_as(ptr)
            self.assertNotEqual(lib.gramgrid_mesh_forward(handle, pointer(x), pointer(out), 256), 0)
            self.assertEqual(lib.gramgrid_mesh_load_weights(handle, pointer(weights), 256), 0, lib.gramgrid_mesh_error())
            self.assertNotEqual(lib.gramgrid_mesh_forward(handle, pointer(x), pointer(out), 255), 0)
            for _ in range(2):
                self.assertEqual(lib.gramgrid_mesh_forward(handle, pointer(x), pointer(out), 256), 0, lib.gramgrid_mesh_error())
                np.testing.assert_array_equal(out, x @ weights)
            self.assertEqual(lib.gramgrid_mesh_weight_loads(handle), 1)
            self.assertEqual(lib.gramgrid_mesh_executions(handle), 2)
        finally:
            lib.gramgrid_mesh_destroy(handle)

if __name__ == "__main__":
    unittest.main()

# Проверенные программные интерфейсы компиляции аппаратного оператора
# Сборка изолирована от служб Lemonade и существующего IRON.
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform_binary


@iron.jit
def add_i32(a: In, b: In, out: Out, *, count: CompileTime[int]):
    tensor_type = np.ndarray[(count,), np.dtype[np.int32]]
    return transform_binary(lambda x, y: x + y, tensor_type, tile_size=16)

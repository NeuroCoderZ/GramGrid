import numpy as np
import aie.iron as iron
from aie.iron import In, Out
from aie.iron.algorithms._pipeline import Stage, pipeline
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo


@iron.jit
def matmul16_i32(a: In, b: In, out: Out):
    tensor = np.ndarray[(256,), np.dtype[np.int32]]
    aa = ObjectFifo(tensor, name='matrix_a')
    bb = ObjectFifo(tensor, name='matrix_b')
    cc = ObjectFifo(tensor, name='matrix_c')

    def body(inputs, outputs, held, constants, iteration):
        for row in range_(16):
            for col in range_(16):
                outputs[0][row * 16 + col] = 0
                for index in range_(16):
                    outputs[0][row * 16 + col] = (
                        outputs[0][row * 16 + col]
                        + inputs[0][row * 16 + index] * inputs[1][index * 16 + col]
                    )

    stage = Stage(body, inputs=[(aa, 1), (bb, 1)], outputs=[cc], iterations=1)
    return pipeline([stage], [tensor, tensor, tensor],
                    [(aa, 'fill', 0), (bb, 'fill', 1), (cc, 'drain', 2)])

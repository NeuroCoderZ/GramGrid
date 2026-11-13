"""Four fixed matrix iterations retain one weight tile inside the NPU loop."""
import numpy as np
import aie.iron as iron
from aie.iron import In, Out
from aie.iron.algorithms._pipeline import Stage, pipeline
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo

@iron.jit
def matmul_sram4(a: In, weights: In, out: Out):
    tensor = np.ndarray[(256,), np.dtype[np.int32]]
    batch = np.ndarray[(1024,), np.dtype[np.int32]]
    aa = ObjectFifo(tensor, name="sram_a")
    bb = ObjectFifo(tensor, name="sram_weights")
    cc = ObjectFifo(tensor, name="sram_c")

    def body(inputs, outputs, held, constants, iteration):
        for row in range_(16):
            for col in range_(16):
                outputs[0][row * 16 + col] = 0
                for k in range_(16):
                    outputs[0][row * 16 + col] = (
                        outputs[0][row * 16 + col]
                        + inputs[0][row * 16 + k] * held[0][k * 16 + col]
                    )

    # The weight consumer is acquired once outside the four-iteration loop.
    stage = Stage(body, inputs=[(aa, 1)], held=[bb], outputs=[cc], iterations=4)
    return pipeline([stage], [batch, tensor, batch],
                    [(aa, "fill", 0), (bb, "fill", 1), (cc, "drain", 2)])

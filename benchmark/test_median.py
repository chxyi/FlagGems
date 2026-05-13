import pytest
import torch

from . import base, consts


class MedianBenchmark(base.UnaryReductionBenchmark):
    """Custom benchmark for median with memory-appropriate shapes."""

    def set_more_shapes(self):
        # Shapes where the reduction dimension N ≤ 4096 (direct kernel limit).
        more_shapes_2d = [(1024, 2**i) for i in range(4, 13, 4)]  # N=16,256,4096
        more_shapes_3d = [(64, 2**i, 64) for i in range(2, 11, 4)]  # N=4,64,1024
        return more_shapes_2d + more_shapes_3d


@pytest.mark.median
def test_median():
    bench = MedianBenchmark(
        op_name="median", torch_op=torch.median, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
